#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neuron_simulator.h"

namespace py = pybind11;

PYBIND11_MODULE(neuron_simulator, m) {
    m.doc() = "Neural Network Simulator with Metabolic Dysfunction";
    
    py::class_<MetabolicCondition>(m, "MetabolicCondition")
        .def(py::init<>())
        .def_readwrite("name", &MetabolicCondition::name)
        .def_readwrite("glucose_level", &MetabolicCondition::glucose_level)
        .def_readwrite("atp_efficiency", &MetabolicCondition::atp_efficiency)
        .def_readwrite("ion_pump_function", &MetabolicCondition::ion_pump_function)
        .def_readwrite("neurotransmitter_synthesis", &MetabolicCondition::neurotransmitter_synthesis)
        .def_readwrite("membrane_integrity", &MetabolicCondition::membrane_integrity)
        .def_readwrite("oxidative_stress", &MetabolicCondition::oxidative_stress)
        .def_readwrite("progressive", &MetabolicCondition::progressive)
        .def_readwrite("onset_timestep", &MetabolicCondition::onset_timestep);
    
    py::class_<StabilityMetrics>(m, "StabilityMetrics")
        .def(py::init<>())
        .def_readwrite("coefficient_of_variation", &StabilityMetrics::coefficient_of_variation)
        .def_readwrite("burst_coefficient", &StabilityMetrics::burst_coefficient)
        .def_readwrite("synchrony_index", &StabilityMetrics::synchrony_index)
        .def_readwrite("entropy", &StabilityMetrics::entropy)
        .def_readwrite("homeostatic_deviation", &StabilityMetrics::homeostatic_deviation)
        .def_readwrite("network_coherence", &StabilityMetrics::network_coherence)
        .def_readwrite("critical_branching_ratio", &StabilityMetrics::critical_branching_ratio);
    
    py::class_<SimulationData>(m, "SimulationData")
        .def(py::init<>())
        .def_readwrite("membrane_potentials", &SimulationData::membrane_potentials)
        .def_readwrite("spike_events", &SimulationData::spike_events)
        .def_readwrite("network_activity", &SimulationData::network_activity)
        .def_readwrite("total_timesteps", &SimulationData::total_timesteps)
        .def_readwrite("total_spikes", &SimulationData::total_spikes);
    
    py::class_<NeuronSimulator>(m, "NeuronSimulator")
        .def(py::init<>())
        .def("run_standard_simulation", &NeuronSimulator::run_standard_simulation,
             "Run standard neural network simulation",
             py::arg("max_timesteps") = 5000)
        .def("run_metabolic_dysfunction_simulation", &NeuronSimulator::run_metabolic_dysfunction_simulation,
             "Run simulation with metabolic dysfunction",
             py::arg("condition"), py::arg("max_timesteps") = 3000)
        .def("run_metabolic_dysfunction_studies", &NeuronSimulator::run_metabolic_dysfunction_studies,
             "Run comprehensive metabolic dysfunction studies")
        .def("create_hypoglycemia", &NeuronSimulator::create_hypoglycemia,
             "Create hypoglycemia metabolic condition")
        .def("create_diabetes_ketoacidosis", &NeuronSimulator::create_diabetes_ketoacidosis,
             "Create diabetes ketoacidosis metabolic condition")
        .def("create_hypoxia", &NeuronSimulator::create_hypoxia,
             "Create hypoxia metabolic condition")
        .def("create_mitochondrial_dysfunction", &NeuronSimulator::create_mitochondrial_dysfunction,
             "Create mitochondrial dysfunction metabolic condition")
        .def("get_simulation_data", &NeuronSimulator::get_simulation_data,
             "Get simulation data")
        .def("calculate_stability_metrics", &NeuronSimulator::calculate_stability_metrics,
             "Calculate stability metrics")
        .def("generate_python_visualization", &NeuronSimulator::generate_python_visualization,
             "Generate Python visualization script")
        .def("export_csv_data", &NeuronSimulator::export_csv_data,
             "Export simulation data to CSV files",
             py::arg("prefix") = "");
}