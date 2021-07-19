# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:13:12 2019

@author: draz
"""
import pandas as pd
import numpy as np
import sys
import math
import os

#sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2019\Python\3.7")
#import powerfactory as pf


class Data:
    def __init__(self):
        self.df_events = pd.DataFrame()
        self.f1_generator_active_powers = pd.DataFrame()
        self.f2_generator_reactive_powers = pd.DataFrame()
        self.f3_out_of_step = pd.DataFrame()
        self.f4_bus_voltage = pd.DataFrame()
        self.f5_bus_angles = pd.DataFrame()
        self.f6_load_active_powers = pd.DataFrame()
        self.f7_load_reactive_powers = pd.DataFrame()
        self.post_fault_results = []
        self.bus_data_post_fault = [[], []]


def set_clearing_time(event, fault_clearing_time__cycles):
    return event.time + fault_clearing_time__cycles * 0.02


def set_fault_location(elements):
    return elements[np.random.randint(len(elements))]


def set_dp(max_load_change):
    return int(np.random.randint(-max_load_change, max_load_change))


def set_fault_rx(event):
    event.i_shc = 0  # three phase short circuit, 1 for two phase and 3 for single phase
    event.R_f = 0.01  # ohm
    event.X_f = 0.01  # ohm
    return event


class Model:
    def __init__(self, grid_model, simulation_time):
        self.app = pf.GetApplication()
        self.events_folder = self.app.GetFromStudyCase('IntEvt')
        self.trafos = self.app.GetCalcRelevantObjects("*.ElmTr2")
        self.loads = self.app.GetCalcRelevantObjects("*.ElmLod")
        self.lines = self.app.GetCalcRelevantObjects("*.ElmLne")
        self.systems = self.app.GetCalcRelevantObjects("*.ElmSym")
        self.buses = self.app.GetCalcRelevantObjects("*.ElmTerm")
        self.initial_conditions = self.app.GetFromStudyCase('ComInc')
        self.sim = self.app.GetFromStudyCase('ComSim')
        self.elm_res = self.app.GetFromStudyCase('Results.ElmRes')
        self.grid_model = grid_model
        self.simulation_time = simulation_time
        self.contingencies = []
        self.event_set = []
        self.Data = Data()

    def model(self):
        # import  PowerFactory  module
        # start PowerFactory  in engine  mode
        self.app.ClearOutputWindow()
        self.app.GetCurrentUser()
        # activate project
        self.app.ActivateProject(self.grid_model)
        #        self.app.Show()
        # Operational library folder
        # open case study folder
        folder_where_study_cases_are_saved = self.app.GetProjectFolder('study')
        # get the contents of the case study folder
        all_study_cases_in_project = folder_where_study_cases_are_saved.GetContents()
        # select case study
        study_case = all_study_cases_in_project[0]
        study_case.Deactivate()
        study_case.Activate()
        # transient simulation
        # initial conditions
        # grid elements

        # load flow
        self.app.GetFromStudyCase("ComLdf")
        # results
        for element in self.elm_res.GetContents():
            element.ClearVars()
        for bus in self.buses:
            self.elm_res.AddVars(bus,
                                 'm:u',  # voltage ('m:u')
                                 'm:phiu',  # angle ('m:phi')
                                 # 'm:fehz',
                                 )

        for sym in self.systems:
            self.elm_res.AddVars(sym,
                                 's:ve',  # p.u    Excitation Voltage
                                 's:pt',  # p.u.   IN    Turbine Power
                                 's:ut',  # p.u.   OUT   Terminal Voltage
                                 's:ie',  # p.u.   OUT   Excitation Current
                                 's:xspeed',  # p.u.   OUT   Speed
                                 's:xme',  # p.u.   OUT   Electrical Torque
                                 's:xmt',  # p.u.   OUT   Mechanical Torque
                                 's:cur1',  # p.u.   OUT   Positive-Sequence Current, Magnitude
                                 's:P1',  # MW     OUT   Positive-Sequence, Active Power
                                 's:Q1',  # Mvar   OUT   Positive-Sequence, Reactive Power
                                 's:outofstep',  # the generator out of step or not
                                 'c:firel',  # deg    Rotor angle with reference to reference machine angle
                                 'c:firot',  # the rotor angel with reference to the nagel of bus volatge
                                 'c:dfrotx',  # the maximum rotor angel difference
                                 'c:fi',  # rotor angel
                                 'c:dorhz',  # speed dilation of the rotor
                                 )
        self.elm_res.AddVars(self.elm_res, 'b:tnow')

    def run_model(self, number_of_events, max_load_change, fault_clearing_time__cycles, path):
        for event in range(number_of_events):
            self.define_events_ieee9buses(max_load_change[0], fault_clearing_time__cycles[0]) if \
                self.grid_model == 'NineBusSystem' else self.define_events_ne39(
                max_load_change[1], fault_clearing_time__cycles[1])

            # configure the initial condition for stability simulation
            self.initial_conditions.iopt_sim = 'rms'  # simulation method 'rms' or 'ins'
            self.initial_conditions.iopt_net = 'sym'  # or 'rst' for unbalanced fault analysis

            self.initial_conditions.iopt_dtgrd = 0.01  # step size (type of the simulation considered: EMT/RMS)
            self.initial_conditions.Execute()  # run the model for the initial condition

            # execute stability simulation
            self.sim.tstop = self.simulation_time  # simulation time in seconds
            self.sim.Execute()  # start grid dynamic simulation
            # export results
            self.export_results(event, path)
            # add the time series of the post fault 
            self.get_post_fault_results(event, path)
            # rest model
            self.rest_calculation()

            #    ldf.iopt_net = 0 #  balanced load
            #    ldf.Execute()

    def set_fault_time(self):
        return np.random.random() * self.simulation_time * 0.5

    def define_events_ieee9buses(self, max_load_change, fault_clearing_time__cycles):
        self.events_folder.CreateObject('EvtShc', 'short circuit')
        self.events_folder.CreateObject('EvtSwitch', 'fault clearing ')
        self.events_folder.CreateObject('EvtLod', 'load event')
        self.event_set = self.events_folder.GetContents()

        event_1 = self.event_set[0]  # which event from the event f
        event_1 = set_fault_rx(event_1)
        event_1.p_target = set_fault_location(self.lines)
        event_1.time = self.set_fault_time()  # execution time of the event

        event_2 = self.event_set[1]
        event_2.p_target = event_1.p_target
        event_2.time = set_clearing_time(event_1, fault_clearing_time__cycles)

        event_3 = self.event_set[2]
        event_3.p_target = set_fault_location(self.loads)
        event_3.time = 0.0  # Isolate the line after 0.5 second
        event_3.iopt_type = 0  # 0 for step change and 1 for ramp change
        event_3.dP = set_dp(max_load_change)
        self.contingencies.extend([event_1, event_2, event_3])

        #    ldf.iopt_net = 0  #balalanced load
        #    ldf.Execute()

    def define_events_ne39(self, max_load_change, fault_clearing_time__cycles):
        self.events_folder.CreateObject('EvtShc', 'short circuit')
        self.events_folder.CreateObject('EvtSwitch', 'fault clearing')
        self.events_folder.CreateObject('EvtSwitch', 'line tripping')
        self.events_folder.CreateObject('EvtLod', 'load event')
        self.event_set = self.events_folder.GetContents()

        event_1 = self.event_set[0]  # which event from the event folder
        event_1.time = self.set_fault_time()  # execution time of the event
        event_1 = set_fault_rx(event_1)
        event_1.p_target = set_fault_location(self.lines)

        event_2 = self.event_set[1]
        event_2.p_target = event_1.p_target
        event_2.time = set_clearing_time(event_1, fault_clearing_time__cycles)

        event_3 = self.event_set[2]
        event_3.p_target = self.lines[
            self.lines.index(event_2.p_target) + 1 if self.lines.index(event_2.p_target) < len(
                self.lines) - 1 else self.lines.index(event_2.p_target) - 1]
        event_3.time = event_2.time + 0.08

        event_4 = self.event_set[3]
        event_4.p_target = set_fault_location(self.loads)
        event_4.time = 0.0  # Isolate the line after 0.5 second
        event_4.iopt_type = 0  # 0 for step change and 1 for ramp change
        event_4.dP = set_dp(max_load_change)
        self.contingencies.extend([event_1, event_2, event_3, event_4])

    def export_results(self, event, path):
        variable = []
        elements = [self.systems[sys_2] for sys_2 in range(len(self.systems))]
        com_res = self.app.GetFromStudyCase('ComRes')
        com_res.iopt_exp = 6  # to export as csv 4 for text file
        com_res.iopt_csel = 0  # 1 only for the selected available and 0 for everything
        com_res.iopt_tsel = 0  #
        com_res.iopt_honly = 0  # export the data with headers
        com_res.iopt_locn = 2
        com_res.iopt_sep = 1  # use the system separator ", or ;"
        com_res.ciopt_head = 1
        com_res.variable = variable
        com_res.element = elements
        com_res.pResult = self.elm_res
        com_res.f_name = os.path.join(path, self.grid_model, 'events', 'event_%s.csv' % event)
        # com_res.f_name = r"C:\DAI_Labor\PhD\grid_model_pf\models\results\NineBusSystem\events\0.csv"
        com_res.Execute()

        # record variables/parameters
        self.Data.f1_generator_active_powers = self.Data.f1_generator_active_powers.append({
            sys_4.loc_name: sys_4.GetAttribute('s:P1') for sys_4 in self.systems}, ignore_index=True)

        self.Data.f2_generator_reactive_powers = self.Data.f2_generator_reactive_powers.append({
            sys_1.loc_name: sys_1.GetAttribute('s:Q1') for sys_1 in self.systems}, ignore_index=True)

        self.Data.f3_out_of_step = self.Data.f3_out_of_step.append({
            sys_3.loc_name: sys_3.GetAttribute('s:outofstep') for sys_3 in self.systems}, ignore_index=True)

        self.Data.f4_bus_voltage = self.Data.f4_bus_voltage.append({
            bus.loc_name: bus.GetAttribute('m:u') for bus in self.buses}, ignore_index=True)

        self.Data.f5_bus_angles = self.Data.f5_bus_angles.append({
            bus.loc_name: bus.GetAttribute('m:phiu') for bus in self.buses}, ignore_index=True)

        self.Data.f6_load_active_powers = self.Data.f6_load_active_powers.append({
            load.loc_name: load.GetAttribute('m:Psum:bus1') for load in self.loads}, ignore_index=True)

        self.Data.f7_load_reactive_powers = self.Data.f7_load_reactive_powers.append({
            load.loc_name: load.GetAttribute('m:Qsum:bus1') for load in self.loads}, ignore_index=True)

        self.Data.df_events = self.Data.df_events.append(
            {'events': [contingency.loc_name for contingency in self.contingencies],
             'event_locations': [contingency.p_target.loc_name for contingency in self.contingencies],
             'event_clearing_time': self.contingencies[1].time,
             'event_time': [contingency.time for contingency in self.contingencies],
             'stability_each_generator': self.Data.f3_out_of_step.values[event].tolist(),
             'system_stability': 'unstable' if 1 in self.Data.f3_out_of_step.values[
                 event].tolist() else 'stable'}, ignore_index=True)

    def get_post_fault_results(self, event, path):

        event_post_fault_data = pd.read_csv(os.path.join(
            path, self.grid_model, 'events', 'event_%s.csv' % event), header=1)

        event_post_fault_data = event_post_fault_data[event_post_fault_data.columns[0: len(self.buses) * 2]].drop(
            list(np.where(event_post_fault_data['b:tnow in s'] < 0)[0]), axis=0).reset_index()

        self.Data.post_fault_results.append(event_post_fault_data)

        event_time = math.ceil(self.contingencies[0].time / self.initial_conditions.iopt_dtgrd)
        event_clearing_time = math.ceil(self.contingencies[1].time / self.initial_conditions.iopt_dtgrd)

        self.Data.bus_data_post_fault[0].append(
            {'bus_data_post_fault_event_%s' % event: event_post_fault_data.iloc[event_time].to_dict()})

        self.Data.bus_data_post_fault[1].append(
            {'bus_data_post_fault_clearing_event_%s' % event: event_post_fault_data.iloc[
                event_clearing_time].to_dict()})

    def rest_calculation(self):
        self.app.ResetCalculation()
        self.contingencies = []
        for event in self.event_set:
            event.Delete()
        self.event_set = []
