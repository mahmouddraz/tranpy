"""Event generation utilities for power system simulations."""

import numpy as np
from typing import List, Optional, Any, Tuple


class EventGenerator:
    """
    Generator for power system contingency events.

    This class provides utilities for creating fault events,
    line tripping, and load changes.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize event generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)

    def generate_fault_location(self, elements: List) -> int:
        """
        Randomly select fault location from available elements.

        Args:
            elements: List of power system elements (lines, buses, etc.)

        Returns:
            Index of selected element
        """
        return self.rng.randint(len(elements))

    def generate_fault_time(self, simulation_time: float, max_fraction: float = 0.5) -> float:
        """
        Generate random fault occurrence time.

        Args:
            simulation_time: Total simulation time in seconds
            max_fraction: Maximum fraction of simulation time for fault

        Returns:
            Fault time in seconds
        """
        return self.rng.random() * simulation_time * max_fraction

    def generate_clearing_time(
        self,
        fault_time: float,
        clearing_cycles: int,
        frequency: float = 50.0
    ) -> float:
        """
        Calculate fault clearing time.

        Args:
            fault_time: Fault occurrence time
            clearing_cycles: Number of cycles to clear fault
            frequency: System frequency in Hz

        Returns:
            Clearing time in seconds
        """
        cycle_duration = 1.0 / frequency
        return fault_time + clearing_cycles * cycle_duration

    def generate_load_change(self, max_change: int) -> int:
        """
        Generate random load change.

        Args:
            max_change: Maximum load change in percentage

        Returns:
            Load change value
        """
        return self.rng.randint(-max_change, max_change)

    def generate_fault_impedance(
        self,
        r_min: float = 0.01,
        r_max: float = 0.01,
        x_min: float = 0.01,
        x_max: float = 0.01
    ) -> tuple:
        """
        Generate fault impedance values.

        Args:
            r_min: Minimum resistance in ohms
            r_max: Maximum resistance in ohms
            x_min: Minimum reactance in ohms
            x_max: Maximum reactance in ohms

        Returns:
            Tuple of (R_f, X_f)
        """
        R_f = self.rng.uniform(r_min, r_max)
        X_f = self.rng.uniform(x_min, x_max)
        return R_f, X_f


class PowerFactoryEventConfigurator:
    """
    Configures PowerFactory-specific event objects.

    This class creates and configures PowerFactory event objects (EvtShc, EvtSwitch, EvtLod)
    for contingency simulations.

    Args:
        events_folder: PowerFactory events folder (IntEvt)
        event_generator: EventGenerator instance for random parameters

    Examples:
        >>> configurator = PowerFactoryEventConfigurator(events_folder, event_gen)
        >>> events = configurator.configure_ieee9bus_events(
        ...     lines, loads, simulation_time, fault_clearing_cycles=10, max_load_change=60
        ... )
    """

    def __init__(self, events_folder: Any, event_generator: EventGenerator):
        self.events_folder = events_folder
        self.event_gen = event_generator
        self.created_events = []

    def create_fault_event(
        self,
        fault_location: Any,
        fault_time: float,
        r_f: float = 0.01,
        x_f: float = 0.01,
        fault_type: int = 0
    ) -> Any:
        """
        Create short circuit event (EvtShc).

        Args:
            fault_location: PowerFactory element where fault occurs
            fault_time: Time when fault occurs (seconds)
            r_f: Fault resistance (ohms)
            x_f: Fault reactance (ohms)
            fault_type: 0=three-phase, 1=two-phase, 3=single-phase

        Returns:
            PowerFactory EvtShc event object
        """
        event = self.events_folder.CreateObject('EvtShc', 'short circuit')
        event.p_target = fault_location
        event.time = fault_time
        event.i_shc = fault_type
        event.R_f = r_f
        event.X_f = x_f

        self.created_events.append(event)
        return event

    def create_clearing_event(
        self,
        fault_event: Any,
        clearing_time: float
    ) -> Any:
        """
        Create fault clearing event (EvtSwitch).

        Args:
            fault_event: The fault event to clear
            clearing_time: Time when fault is cleared (seconds)

        Returns:
            PowerFactory EvtSwitch event object
        """
        event = self.events_folder.CreateObject('EvtSwitch', 'fault clearing')
        event.p_target = fault_event.p_target
        event.time = clearing_time

        self.created_events.append(event)
        return event

    def create_line_trip_event(
        self,
        line: Any,
        trip_time: float
    ) -> Any:
        """
        Create line tripping event (EvtSwitch).

        Args:
            line: PowerFactory line element to trip
            trip_time: Time when line is tripped (seconds)

        Returns:
            PowerFactory EvtSwitch event object
        """
        event = self.events_folder.CreateObject('EvtSwitch', 'line tripping')
        event.p_target = line
        event.time = trip_time

        self.created_events.append(event)
        return event

    def create_load_event(
        self,
        load: Any,
        load_time: float = 0.0,
        load_change_percent: int = 0,
        change_type: int = 0
    ) -> Any:
        """
        Create load change event (EvtLod).

        Args:
            load: PowerFactory load element
            load_time: Time when load changes (seconds)
            load_change_percent: Load change in percentage
            change_type: 0=step change, 1=ramp change

        Returns:
            PowerFactory EvtLod event object
        """
        event = self.events_folder.CreateObject('EvtLod', 'load event')
        event.p_target = load
        event.time = load_time
        event.iopt_type = change_type
        event.dP = load_change_percent

        self.created_events.append(event)
        return event

    def configure_ieee9bus_events(
        self,
        lines: List[Any],
        loads: List[Any],
        simulation_time: float,
        fault_clearing_cycles: int,
        max_load_change: int,
        frequency: float = 50.0
    ) -> Tuple[List[Any], dict]:
        """
        Configure events for IEEE 9-bus system (3 events).

        Event sequence:
        1. Short circuit on random line
        2. Fault clearing
        3. Load change on random load

        Args:
            lines: List of PowerFactory line elements
            loads: List of PowerFactory load elements
            simulation_time: Total simulation time
            fault_clearing_cycles: Cycles to clear fault
            max_load_change: Maximum load change percentage
            frequency: System frequency (Hz)

        Returns:
            Tuple of (event_list, event_info_dict)
        """
        # Generate random parameters
        fault_line_idx = self.event_gen.generate_fault_location(lines)
        fault_line = lines[fault_line_idx]

        fault_time = self.event_gen.generate_fault_time(simulation_time)
        clearing_time = self.event_gen.generate_clearing_time(
            fault_time, fault_clearing_cycles, frequency
        )

        r_f, x_f = self.event_gen.generate_fault_impedance()

        load_idx = self.event_gen.generate_fault_location(loads)
        load = loads[load_idx]
        load_change = self.event_gen.generate_load_change(max_load_change)

        # Create events
        event_1 = self.create_fault_event(fault_line, fault_time, r_f, x_f)
        event_2 = self.create_clearing_event(event_1, clearing_time)
        event_3 = self.create_load_event(load, 0.0, load_change, change_type=0)

        events = [event_1, event_2, event_3]

        # Event metadata
        event_info = {
            'event_types': [e.loc_name for e in events],
            'event_locations': [e.p_target.loc_name for e in events],
            'event_times': [e.time for e in events],
            'fault_time': fault_time,
            'clearing_time': clearing_time,
            'fault_impedance': {'R_f': r_f, 'X_f': x_f},
            'load_change': load_change
        }

        return events, event_info

    def configure_newengland39_events(
        self,
        lines: List[Any],
        loads: List[Any],
        simulation_time: float,
        fault_clearing_cycles: int,
        max_load_change: int,
        frequency: float = 50.0
    ) -> Tuple[List[Any], dict]:
        """
        Configure events for New England 39-bus system (4 events).

        Event sequence:
        1. Short circuit on random line
        2. Fault clearing
        3. Adjacent line tripping
        4. Load change on random load

        Args:
            lines: List of PowerFactory line elements
            loads: List of PowerFactory load elements
            simulation_time: Total simulation time
            fault_clearing_cycles: Cycles to clear fault
            max_load_change: Maximum load change percentage
            frequency: System frequency (Hz)

        Returns:
            Tuple of (event_list, event_info_dict)
        """
        # Generate random parameters
        fault_line_idx = self.event_gen.generate_fault_location(lines)
        fault_line = lines[fault_line_idx]

        fault_time = self.event_gen.generate_fault_time(simulation_time)
        clearing_time = self.event_gen.generate_clearing_time(
            fault_time, fault_clearing_cycles, frequency
        )

        r_f, x_f = self.event_gen.generate_fault_impedance()

        # Select adjacent line for tripping (avoid index errors)
        if fault_line_idx < len(lines) - 1:
            trip_line = lines[fault_line_idx + 1]
        else:
            trip_line = lines[fault_line_idx - 1]

        trip_time = clearing_time + 0.08  # Trip 80ms after clearing

        load_idx = self.event_gen.generate_fault_location(loads)
        load = loads[load_idx]
        load_change = self.event_gen.generate_load_change(max_load_change)

        # Create events
        event_1 = self.create_fault_event(fault_line, fault_time, r_f, x_f)
        event_2 = self.create_clearing_event(event_1, clearing_time)
        event_3 = self.create_line_trip_event(trip_line, trip_time)
        event_4 = self.create_load_event(load, 0.0, load_change, change_type=0)

        events = [event_1, event_2, event_3, event_4]

        # Event metadata
        event_info = {
            'event_types': [e.loc_name for e in events],
            'event_locations': [e.p_target.loc_name for e in events],
            'event_times': [e.time for e in events],
            'fault_time': fault_time,
            'clearing_time': clearing_time,
            'fault_impedance': {'R_f': r_f, 'X_f': x_f},
            'load_change': load_change
        }

        return events, event_info

    def clear_created_events(self):
        """Delete all events created by this configurator."""
        for event in self.created_events:
            event.Delete()
        self.created_events = []
