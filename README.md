# Human Neuron Network Simulation

A comprehensive Python-based neural network simulator designed for studying metabolic dysfunction effects on neuronal activity and network behavior. This simulator models biologically realistic neurons and allows researchers to investigate how various metabolic conditions affect neural network dynamics.

![Demo](https://github.com/xalentis/PyNeurolib/blob/master/demo.png)

Includes a number of mental health conditions:

![Demo](https://github.com/xalentis/PyNeurolib/blob/master/demo2.png)

## Features

### Core Simulation Capabilities
- **Standard Network Simulation**: Model healthy neural network behavior with realistic membrane dynamics
- **Metabolic Dysfunction Studies**: Comprehensive analysis of pathological conditions affecting neural function
- **Single Condition Testing**: Focused investigation of specific metabolic disorders
- **Real-time Visualization**: Interactive plotting of simulation results with detailed network metrics

### Supported Metabolic Conditions
- **Severe Hypoglycemia**: Low glucose levels affecting cellular energy production
- **Diabetic Ketoacidosis**: Metabolic acidosis with altered glucose metabolism
- **Cerebral Hypoxia**: Reduced oxygen availability affecting ATP production
- **Mitochondrial Dysfunction**: Impaired cellular energy production mechanisms

### Analysis Metrics
- Membrane potential dynamics
- Spike raster patterns
- Network activity analysis
- Stability metrics including:
  - Coefficient of variation
  - Synchrony index
  - Network entropy
  - Homeostatic deviation

## Installation

### Prerequisites
- Python 3.7+
- Required packages:
```bash
pip install numpy pandas matplotlib
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/neuron-simulator.git
cd neuron-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Simulator
Execute the main simulation interface:
```bash
python simulator.py
```

### Interactive Menu Options

#### 1. Standard Network Simulation
Runs a baseline simulation of healthy neural network activity:
- 5000 timesteps of simulation
- Exports CSV data for analysis
- Generates comprehensive visualization
- Calculates stability and synchrony metrics

#### 2. Metabolic Dysfunction Study
Comprehensive analysis across all supported metabolic conditions:
- Runs simulations for all four metabolic conditions
- Generates comparative visualizations
- Exports condition-specific datasets

#### 3. Single Metabolic Condition Test
Focused study of individual metabolic disorders:
- Interactive condition selection
- Detailed parameter display
- Clinical interpretation of results
- Condition-specific data export

### Programmatic Usage

```python
import neuron_simulator

# Create simulator instance
simulator = neuron_simulator.NeuronSimulator()

# Run standard simulation
simulator.run_standard_simulation(5000)

# Create and test specific condition
condition = simulator.create_hypoglycemia()
simulator.run_metabolic_dysfunction_simulation(condition, 3000)

# Export results
simulator.export_csv_data()
metrics = simulator.calculate_stability_metrics()
```

## Output Files

The simulator generates several output files for analysis:

### CSV Data Files
- `membrane_potentials.csv`: Time-series membrane potential data for each neuron
- `spike_raster.csv`: Spike timing data with neuron IDs and timestamps
- `activity_summary.csv`: Network-level activity metrics over time

### Visualization Files
- `simulation_results.png`: Comprehensive 4-panel visualization including:
  - Membrane potential traces
  - Spike raster plot
  - Network activity over time
  - Spike rate analysis

### Condition-Specific Files
For metabolic condition studies, files are prefixed with condition names:
- `Severe_Hypoglycemia_membrane_potentials.csv`
- `Diabetic_Ketoacidosis_simulation_results.png`
- etc.

## Clinical Interpretation

The simulator provides automated clinical interpretation of results:

- **SEVERE DYSFUNCTION**: Spike rate < 0.1 spikes/timestep
  - Network activity critically reduced
  - Indicates potential neuronal death or severe metabolic compromise

- **HYPEREXCITABILITY**: Spike rate > 0.5 spikes/timestep  
  - Seizure-like network activity
  - May indicate metabolic excitotoxicity

- **MILD DYSFUNCTION**: Moderate spike rates
  - Network showing partial compensation
  - Early-stage metabolic effects

## Technical Details

### Simulation Parameters
- Default simulation length: 3000-5000 timesteps
- Multiple neuron network architecture
- Biologically realistic membrane dynamics
- Metabolic parameter modulation including:
  - Glucose levels (mg/dL)
  - ATP efficiency (%)
  - Ion pump function (%)
  - Dysfunction onset timing

### Metrics Calculations
- **Coefficient of Variation**: Measure of spike timing regularity
- **Synchrony Index**: Degree of network synchronization
- **Network Entropy**: Information-theoretic measure of activity patterns
- **Homeostatic Deviation**: Membrane potential stability measure

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-condition`)
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

### Adding New Metabolic Conditions
To add new metabolic conditions:
1. Implement condition creation method in `neuron_simulator` module
2. Add condition to the menu system in `simulator.py`
3. Update documentation and tests

## Dependencies

- `numpy`: Numerical computations and array operations
- `pandas`: Data manipulation and CSV export/import
- `matplotlib`: Visualization and plotting
- `neuron_simulator`: Core simulation engine (custom module)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{neuron_simulator,
  title={Human Neuron Network Simulation},
  author={Gideon Vos},
  year={2025},
  url={https://github.com/yourusername/neuron-simulator}
}
```

## Support

For questions, bug reports, or feature requests, please open an issue on the GitHub repository.

## Acknowledgments

This simulator was developed to support research into metabolic effects on neural network dynamics and may be useful for:
- Computational neuroscience research
- Metabolic disorder modeling
- Educational demonstrations of neural network behavior
- Drug effect simulation studies
