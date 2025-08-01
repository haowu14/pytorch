# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional
from unittest import skipIf, skipUnless
from unittest.mock import mock_open, patch

import torch
from torch.distributed.numa.binding import (
    _get_set_of_int_from_ranges_str,
    AffinityMode,
    NumaOptions,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _dummy_test_function():
    """A module-level function for testing picklability."""
    return "hello"


@dataclass(frozen=True)
class MockDeviceProperties:
    name: str
    major: int
    minor: int
    total_memory: str
    multi_processor_count: int
    uuid: str
    pci_bus_id: int
    pci_device_id: int
    pci_domain_id: int
    L2_cache_size: str


_real_open = open


@skipIf(
    sys.platform == "win32",
    "Windows is missing various os module attributes like sched_getaffinity",
)
@skipUnless(
    torch.distributed.is_available(), "Need access to some distributed submodules"
)
class NumaBindingTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self._mock_file_path_to_contents: dict[str, str] = {}
        self._mock_device_properties: list[MockDeviceProperties] = []
        self._mock_num_logical_cpus = 0
        self._mock_num_numa_nodes = 0
        self._mock_num_sockets = 0

        self._context_managers_to_apply_to_all_tests = [
            patch("torch.cuda.device_count", self._mock_device_count),
            patch("torch.cuda.get_device_properties", self._mock_get_device_properties),
            patch("torch.cuda.is_available", self._mock_is_available),
            patch("builtins.open", new=self._mock_open),
            patch("os.listdir", new=self._mock_listdir),
            patch("os.sched_getaffinity", new=self._mock_sched_getaffinity),
            patch("shutil.which", return_value="/usr/bin/numactl"),
            patch("subprocess.run"),
        ]

        for context_manager in self._context_managers_to_apply_to_all_tests:
            context_manager.__enter__()

    def tearDown(self) -> None:
        for context_manager in self._context_managers_to_apply_to_all_tests:
            context_manager.__exit__(None, None, None)
        super().tearDown()

    def _add_mock_hardware(
        self,
        *,
        num_sockets: int,
        num_numa_nodes_per_socket: int,
        num_gpus_per_numa_node: int,
        num_l3_caches_per_numa_node: int,
        num_physical_core_per_l3_cache: int,
    ) -> None:
        """
        It's not fun, but we mock everything down to sysfs level
        to make sure we get really thorough coverage.
        """
        for socket_index in range(num_sockets):
            for numa_node_index in range(
                self._mock_num_numa_nodes,
                self._mock_num_numa_nodes + num_numa_nodes_per_socket,
            ):
                self._mock_file_contents(
                    file_path=f"/sys/devices/system/node/node{numa_node_index}/cpulist",
                    contents=f"{self._mock_num_logical_cpus}-"
                    + f"{self._mock_num_logical_cpus + num_l3_caches_per_numa_node * num_physical_core_per_l3_cache * 2 - 1}",
                )
                for gpu_index in range(
                    len(self._mock_device_properties),
                    len(self._mock_device_properties) + num_gpus_per_numa_node,
                ):
                    device_properties = MockDeviceProperties(
                        name=f"mock_gpu_{gpu_index}",
                        major=8,
                        minor=0,
                        total_memory="512GB",
                        multi_processor_count=256,
                        uuid=f"mock_gpu_uuid_{gpu_index}",
                        pci_bus_id=gpu_index,
                        pci_device_id=gpu_index,
                        pci_domain_id=gpu_index,
                        L2_cache_size="40MB",
                    )
                    self._mock_device_properties.append(device_properties)
                    pci_numa_node_path = (
                        self._get_corresponding_pci_numa_node_file_path(
                            device_properties=device_properties
                        )
                    )
                    self._mock_file_contents(
                        file_path=pci_numa_node_path,
                        contents=str(numa_node_index),
                    )

                for _ in range(num_l3_caches_per_numa_node):
                    lowest_logical_cpu_index_on_l3 = self._mock_num_logical_cpus
                    highest_logical_cpu_index_on_l3 = (
                        self._mock_num_logical_cpus
                        + 2 * num_physical_core_per_l3_cache
                        - 1
                    )
                    for logical_cpu_index in range(
                        self._mock_num_logical_cpus,
                        self._mock_num_logical_cpus
                        # Assume hyperthreaded
                        + 2 * num_physical_core_per_l3_cache,
                    ):
                        thread_siblings_range_str = (
                            f"{logical_cpu_index - 1}-{logical_cpu_index}"
                            if logical_cpu_index % 2
                            else f"{logical_cpu_index}-{logical_cpu_index + 1}"
                        )
                        self._mock_file_contents(
                            file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/thread_siblings_list",
                            contents=thread_siblings_range_str,
                        )
                        # Unrelated file our logic should know to skip
                        self._mock_file_contents(
                            file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/paulwuzhere",
                            contents="Data",
                        )
                        self._mock_file_contents(
                            file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/physical_package_id",
                            contents=str(socket_index),
                        )
                        for cache_level in range(5):
                            self._mock_file_contents(
                                file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/index{cache_level}/type",
                                contents="ShouldSkip" if cache_level == 4 else "Data",
                            )
                            self._mock_file_contents(
                                file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/index{cache_level}/level",
                                contents=str(cache_level),
                            )
                            self._mock_file_contents(
                                file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/index{cache_level}/shared_cpu_list",
                                contents=(
                                    f"{lowest_logical_cpu_index_on_l3}-{highest_logical_cpu_index_on_l3}"
                                    if cache_level == 3
                                    # Assume L1-2 are per physical core
                                    else thread_siblings_range_str
                                ),
                            )
                        self._mock_num_logical_cpus += 1
                self._mock_num_numa_nodes += 1
            self._mock_num_sockets += 1
        self._mock_file_contents(
            file_path="/sys/devices/system/node/possible",
            contents=f"0-{self._mock_num_numa_nodes - 1}",
        )

    def _mock_is_available(self) -> bool:
        return len(self._mock_device_properties) > 0

    def _get_corresponding_pci_numa_node_file_path(
        self, *, device_properties: MockDeviceProperties
    ) -> str:
        pci_addr = (
            f"{device_properties.pci_domain_id:04x}:"
            + f"{device_properties.pci_bus_id:02x}:{device_properties.pci_device_id:02x}.0"
        )
        return f"/sys/bus/pci/devices/{pci_addr}/numa_node"

    def _mock_file_contents(self, *, file_path: str, contents: str) -> None:
        self._mock_file_path_to_contents[file_path] = contents

    def _mock_device_count(self) -> int:
        return len(self._mock_device_properties)

    def _mock_get_device_properties(self, index: int) -> MockDeviceProperties:
        return self._mock_device_properties[index]

    def _mock_open(self, path, *args, **kwargs) -> Any:
        # Handle file descriptors (integers) used by multiprocessing
        if isinstance(path, int):
            return _real_open(path, *args, **kwargs)

        # Handle file paths (strings)
        if isinstance(path, str):
            if path in self._mock_file_path_to_contents:
                return mock_open(read_data=self._mock_file_path_to_contents[path])()
            if path.startswith("/sys/"):
                raise FileNotFoundError(f"File {path} was not mocked.")

        # Fallback to real open for other cases
        return _real_open(path, *args, **kwargs)

    def _mock_listdir(self, target_path: str) -> set[str]:
        if not target_path.endswith("/"):
            target_path += "/"
        return {
            mock_path.split(target_path)[1].split("/")[0]
            for mock_path in self._mock_file_path_to_contents
            if mock_path.startswith(target_path)
        }

    def _mock_sched_getaffinity(self, pid: int) -> set[int]:
        return set(range(self._mock_num_logical_cpus))

    def _start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
        self, *, numa_options: Optional[NumaOptions], local_rank: int
    ) -> Optional[set[int]]:
        """
        Tests the NUMA binding logic by calling the wrapper functions directly.
        This provides good coverage of the entrypoint wrapping logic without
        the complexity of actual multiprocessing in a heavily mocked environment.
        """
        if numa_options is None:
            return None

        captured_cpu_indices = None

        def mock_sched_setaffinity_noop(pid: int, cpu_indices: set[int]) -> None:
            nonlocal captured_cpu_indices
            # Only capture if this is the current process (pid=0)
            if pid == 0:
                captured_cpu_indices = cpu_indices
            # Function as a no-op - don't actually set affinity

        # Import the wrapper function to test it directly
        from torch.distributed.numa.binding import (
            maybe_wrap_entrypoint_with_numa_bindings,
        )

        # Create the wrapped entrypoint
        wrapped_entrypoint = maybe_wrap_entrypoint_with_numa_bindings(
            "echo", numa_options
        )

        # Set up the environment as it would be in a real process
        old_local_rank = os.environ.get("LOCAL_RANK")
        os.environ["LOCAL_RANK"] = str(local_rank)

        try:
            with patch("os.sched_setaffinity", mock_sched_setaffinity_noop):
                # Call the wrapped entrypoint directly - this tests the actual production path
                # The wrapped entrypoint will call apply_numa_bindings_to_current_process
                try:
                    wrapped_entrypoint("Hello, world!")
                except SystemExit:
                    # os.execvp will cause SystemExit in the test environment, which is expected
                    pass
        finally:
            # Restore the original LOCAL_RANK
            if old_local_rank is not None:
                os.environ["LOCAL_RANK"] = old_local_rank
            elif "LOCAL_RANK" in os.environ:
                del os.environ["LOCAL_RANK"]

        return captured_cpu_indices

    def test_node_numa_binding(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.NODE), local_rank=11
            )
        )
        # There are 8 numa nodes and 2 GPUs per numa node, so GPU 11 would be
        # on numa node 11 // 2 = 5.
        # Numa node 5 has CPUs 80-95 (5 * 16 to (5+1) * 16 - 1)
        expected_cpu_indices = set(range(80, 96))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_no_numa_binding_if_numa_options_not_provided(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=None, local_rank=11
            )
        )
        # Should be None since no NUMA binding was applied
        self.assertIsNone(cpu_indices)

    def test_default_numa_binding(self) -> None:
        # Inner import to avoid crashing if not torch.distributed.is_available()
        from torch.distributed.launcher.api import LaunchConfig

        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with patch(
            "torch.distributed.launcher.api.get_default_numa_options",
            return_value=NumaOptions(
                affinity_mode=AffinityMode.NODE, should_fall_back_if_binding_fails=True
            ),
        ):
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=1,
                # Don't provide numa_options
            )
        self.assertEqual(
            launch_config.numa_options,
            NumaOptions(
                affinity_mode=AffinityMode.NODE, should_fall_back_if_binding_fails=True
            ),
        )

    def test_explicit_numa_options_overrides_default(self) -> None:
        # Inner import to avoid crashing if not torch.distributed.is_available()
        from torch.distributed.launcher.api import LaunchConfig

        with patch(
            "torch.distributed.launcher.api.get_default_numa_options",
            return_value=NumaOptions(affinity_mode=AffinityMode.NODE),
        ):
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=1,
                numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
            )
        self.assertEqual(
            launch_config.numa_options,
            NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
        )

    def test_socket_numa_binding_with_multiple_numa_per_socket(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.SOCKET),
                local_rank=15,
            )
        )
        # GPU 15 is on numa node 15 // 2 = 7, which is on socket 7 // 2 = 3
        # Socket 3 has numa nodes 6-7, which have CPUs 96-127
        expected_cpu_indices = set(range(96, 128))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_socket_numa_binding_with_single_numa_per_socket(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.SOCKET),
                local_rank=7,
            )
        )
        # GPU 7 is on numa node 7 // 2 = 3, which is socket 3
        # Socket 3 has numa node 3, which has CPUs 48-63 (3 * 16 to (3+1) * 16 - 1)
        expected_cpu_indices = set(range(48, 64))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_exclusive_numa_binding(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=3,
        )

        cpu_indices_0 = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
                local_rank=0,
            )
        )
        # GPU 0 gets an extra physical core due to odd number of physical cores on numa node
        # Should get CPUs 0-3 (2 physical cores * 2 threads each)
        expected_cpu_indices_0 = set(range(0, 4))
        self.assertEqual(cpu_indices_0, expected_cpu_indices_0)

        cpu_indices_1 = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
                local_rank=1,
            )
        )
        # GPU 1 gets the remaining physical core
        # Should get CPUs 4-5 (1 physical core * 2 threads)
        expected_cpu_indices_1 = set(range(4, 6))
        self.assertEqual(cpu_indices_1, expected_cpu_indices_1)

    def test_exclusive_raises_if_too_few_physical_cores(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "There are only 1 physical cores on numa_node_index=0, but there are 2 GPUs associated with this NUMA node.",
        ):
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
                local_rank=1,
            )

    def test_core_complex_numa_binding_with_extra_l3(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=3,
            num_physical_core_per_l3_cache=3,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
                local_rank=3,
            )
        )
        # GPU 3 is on numa node 3 // 2 = 1, and gets the second L3 cache (index 1)
        # The second L3 on the second numa node should have CPUs 24-29
        expected_cpu_indices = set(range(24, 30))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_core_complex_numa_binding_with_fewer_l3_than_gpu(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=3,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
                local_rank=3,
            )
        )
        # GPU 3 is on numa node 3 // 2 = 1, and there's only 1 L3 cache per numa node
        # So GPU 3 shares the same cores as GPU 2, which are CPUs 6-11
        expected_cpu_indices = set(range(6, 12))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_core_complex_prefers_caches_with_more_cpus(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=3,
            num_physical_core_per_l3_cache=3,
        )

        # Only some subset of the CPUs are available this time.
        with patch("os.sched_getaffinity", return_value={0, 4, 6, 7, 9}):
            cpu_indices = self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
                local_rank=0,
            )

        # Should bind to the second L3 because it has the most available CPUs (6, 7, 9)
        expected_cpu_indices = {6, 7, 9}
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_core_complex_tiebreak_prefers_lower_cache_key(self) -> None:
        """
        When several max‑level caches expose the same number of logical CPUs,
        prioritize binding to caches with lower cpu indices first.
        """
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=2,
            num_physical_core_per_l3_cache=1,
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
                local_rank=0,
            )
        )
        # Should prefer the first L3 cache with CPUs 0-1
        expected_cpu_indices = set(range(0, 2))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_binds_to_node_0_if_node_stored_as_minus_one(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        device_0_properties = self._mock_get_device_properties(0)
        # Overwrite the existing mock file
        self._mock_file_contents(
            file_path=self._get_corresponding_pci_numa_node_file_path(
                device_properties=device_0_properties
            ),
            contents="-1",
        )

        cpu_indices = (
            self._start_test_processes_and_get_sched_setaffinity_args_for_local_rank(
                numa_options=NumaOptions(affinity_mode=AffinityMode.NODE), local_rank=0
            )
        )
        # Should bind to node 0 CPUs (0-1)
        expected_cpu_indices = set(range(0, 2))
        self.assertEqual(cpu_indices, expected_cpu_indices)

    def test_fallback(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        def mock_sched_setaffinity_that_fails(pid: int, cpu_indices: set[int]) -> None:
            raise OSError("Mock failure in sched_setaffinity")

        # Import here to avoid circular imports
        from torch.distributed.numa.binding import (
            apply_numa_bindings_to_current_process,
        )

        with (
            patch("torch.distributed.numa.binding.signpost_event") as signpost_patch,
            patch(
                "os.sched_setaffinity", side_effect=mock_sched_setaffinity_that_fails
            ),
        ):
            # Call apply_numa_bindings_to_current_process directly to test fallback
            apply_numa_bindings_to_current_process(
                numa_options=NumaOptions(
                    affinity_mode=AffinityMode.NODE,
                    should_fall_back_if_binding_fails=True,
                ),
                gpu_index=0,
            )

        # Check that signpost_event was called with exception information
        self.assertTrue(signpost_patch.called)
        call_args = signpost_patch.call_args
        self.assertEqual(call_args.kwargs["category"], "numa_binding")
        self.assertEqual(call_args.kwargs["name"], "wrap_command_exception")
        self.assertIn("traceback", call_args.kwargs["parameters"])
        self.assertIn("OSError", call_args.kwargs["parameters"]["traceback"])

    def test_get_set_of_int_from_ranges_str(self) -> None:
        self.assertEqual(
            _get_set_of_int_from_ranges_str("0-2,4,6-7"), {0, 1, 2, 4, 6, 7}
        )

    def test_entrypoint_wrappers_are_picklable(self) -> None:
        """Test that our entrypoint wrappers can be pickled (required for multiprocessing)."""
        import pickle

        from torch.distributed.numa.binding import (
            maybe_wrap_entrypoint_with_numa_bindings,
        )

        # Test string entrypoint wrapper
        numa_options = NumaOptions(affinity_mode=AffinityMode.NODE)
        str_wrapper = maybe_wrap_entrypoint_with_numa_bindings("echo", numa_options)

        # Should be able to pickle and unpickle
        pickled_str_wrapper = pickle.dumps(str_wrapper)
        unpickled_str_wrapper = pickle.loads(pickled_str_wrapper)
        self.assertIsNotNone(unpickled_str_wrapper)

        # Test callable entrypoint wrapper using a module-level function
        callable_wrapper = maybe_wrap_entrypoint_with_numa_bindings(
            _dummy_test_function, numa_options
        )

        # Should be able to pickle and unpickle
        pickled_callable_wrapper = pickle.dumps(callable_wrapper)
        unpickled_callable_wrapper = pickle.loads(pickled_callable_wrapper)
        self.assertIsNotNone(unpickled_callable_wrapper)


if __name__ == "__main__":
    run_tests()
