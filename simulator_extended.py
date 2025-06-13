import neuron_simulator
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy

random.seed(43)
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def print_menu():
    print("=== Enhanced Human Neuron Network Simulation ===")
    print("Choose simulation mode:")
    print("1. Standard Network Simulation")
    print("2. Metabolic Dysfunction Study")
    print("3. Mental Health Conditions Study")
    print("4. Single Condition Test")
    print("5. Comparative Analysis")
    print("6. Long-term Progression Study")
    print("7. Exit")


def create_mental_health_conditions():
    conditions = {}
    
    # Major Depression
    depression = neuron_simulator.MetabolicCondition()
    depression.name = "Major Depression"
    depression.glucose_level = 85.0  # Slightly reduced
    depression.atp_efficiency = 0.8  # Reduced energy metabolism
    depression.ion_pump_function = 0.85  # Slightly impaired
    depression.neurotransmitter_synthesis = 0.4  # Severely reduced (serotonin/dopamine)
    depression.membrane_integrity = 0.9
    depression.oxidative_stress = 0.3  # Increased oxidative stress
    depression.progressive = True
    depression.onset_timestep = 1000
    conditions['depression'] = depression
    
    # Bipolar Disorder - Manic Phase
    bipolar_manic = neuron_simulator.MetabolicCondition()
    bipolar_manic.name = "Bipolar Disorder (Manic)"
    bipolar_manic.glucose_level = 110.0  # Elevated
    bipolar_manic.atp_efficiency = 1.3  # Hypermetabolism
    bipolar_manic.ion_pump_function = 1.2  # Overactive
    bipolar_manic.neurotransmitter_synthesis = 1.8  # Excess dopamine/norepinephrine
    bipolar_manic.membrane_integrity = 0.95
    bipolar_manic.oxidative_stress = 0.4  # High due to hyperactivity
    bipolar_manic.progressive = False
    bipolar_manic.onset_timestep = 500
    conditions['bipolar_manic'] = bipolar_manic
    
    # Bipolar Disorder - Depressive Phase
    bipolar_depressive = neuron_simulator.MetabolicCondition()
    bipolar_depressive.name = "Bipolar Disorder (Depressive)"
    bipolar_depressive.glucose_level = 75.0  # Low
    bipolar_depressive.atp_efficiency = 0.6  # Very low energy
    bipolar_depressive.ion_pump_function = 0.7  # Significantly impaired
    bipolar_depressive.neurotransmitter_synthesis = 0.3  # Severely reduced
    bipolar_depressive.membrane_integrity = 0.85
    bipolar_depressive.oxidative_stress = 0.5  # High
    bipolar_depressive.progressive = False
    bipolar_depressive.onset_timestep = 500
    conditions['bipolar_depressive'] = bipolar_depressive
    
    # Schizophrenia
    schizophrenia = neuron_simulator.MetabolicCondition()
    schizophrenia.name = "Schizophrenia"
    schizophrenia.glucose_level = 95.0
    schizophrenia.atp_efficiency = 0.7  # Impaired cellular energy
    schizophrenia.ion_pump_function = 0.8  # Dysregulated ion channels
    schizophrenia.neurotransmitter_synthesis = 1.5  # Excess dopamine, reduced GABA
    schizophrenia.membrane_integrity = 0.8  # Compromised membrane function
    schizophrenia.oxidative_stress = 0.6  # High oxidative stress
    schizophrenia.progressive = True
    schizophrenia.onset_timestep = 2000
    conditions['schizophrenia'] = schizophrenia
    
    # Anxiety Disorder
    anxiety = neuron_simulator.MetabolicCondition()
    anxiety.name = "Anxiety Disorder"
    anxiety.glucose_level = 105.0  # Elevated due to stress
    anxiety.atp_efficiency = 0.9
    anxiety.ion_pump_function = 1.1  # Hyperexcitability
    anxiety.neurotransmitter_synthesis = 0.6  # Reduced GABA, excess glutamate
    anxiety.membrane_integrity = 0.95
    anxiety.oxidative_stress = 0.4  # Stress-induced
    anxiety.progressive = False
    anxiety.onset_timestep = 200
    conditions['anxiety'] = anxiety
    
    # ADHD
    adhd = neuron_simulator.MetabolicCondition()
    adhd.name = "ADHD"
    adhd.glucose_level = 100.0
    adhd.atp_efficiency = 0.95
    adhd.ion_pump_function = 0.9  # Slightly dysregulated
    adhd.neurotransmitter_synthesis = 0.7  # Reduced dopamine/norepinephrine
    adhd.membrane_integrity = 0.95
    adhd.oxidative_stress = 0.2
    adhd.progressive = False
    adhd.onset_timestep = 100
    conditions['adhd'] = adhd
    
    # PTSD
    ptsd = neuron_simulator.MetabolicCondition()
    ptsd.name = "PTSD"
    ptsd.glucose_level = 95.0
    ptsd.atp_efficiency = 0.8  # Stress-impaired metabolism
    ptsd.ion_pump_function = 0.85
    ptsd.neurotransmitter_synthesis = 0.5  # Dysregulated stress response
    ptsd.membrane_integrity = 0.9
    ptsd.oxidative_stress = 0.7  # Very high due to chronic stress
    ptsd.progressive = True
    ptsd.onset_timestep = 1500
    conditions['ptsd'] = ptsd
    return conditions


def calculate_advanced_metrics(membrane_data, spike_data, activity_data):
    metrics = {}
    
    # Convert membrane potentials to numpy array
    potentials = membrane_data.iloc[:, 1:].values
    
    # Network synchronization
    correlations = np.corrcoef(potentials.T)
    metrics['synchronization'] = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
    
    # Oscillatory activity (simulate different frequency bands)
    if len(activity_data) > 100:
        signal_data = activity_data['Average_Potential'].values
        
        # Gamma band (30-100 Hz simulation)
        gamma_power = np.var(signal_data[::2] - signal_data[1::2])
        metrics['gamma_power'] = gamma_power
        
        # Beta band (13-30 Hz simulation)
        beta_signal = signal_data[::4]
        metrics['beta_power'] = np.var(beta_signal) if len(beta_signal) > 10 else 0
        
        # Alpha band (8-13 Hz simulation)
        alpha_signal = signal_data[::8]
        metrics['alpha_power'] = np.var(alpha_signal) if len(alpha_signal) > 10 else 0
    
    # Neural complexity (approximate entropy)
    if len(activity_data) > 50:
        signal_data = activity_data['Average_Potential'].values
        # Discretize signal for entropy calculation
        bins = np.linspace(signal_data.min(), signal_data.max(), 10)
        digitized = np.digitize(signal_data, bins)
        metrics['neural_entropy'] = entropy(np.bincount(digitized))
    
    # Spike irregularity
    if len(spike_data) > 0:
        spike_times = spike_data.groupby('Neuron_ID')['Timestep'].apply(list)
        isis = []  # Inter-spike intervals
        for neuron_spikes in spike_times:
            if len(neuron_spikes) > 1:
                isis.extend(np.diff(neuron_spikes))
        if isis:
            metrics['spike_irregularity'] = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        else:
            metrics['spike_irregularity'] = 0
    
    # Burst detection
    if len(spike_data) > 0:
        spike_counts = spike_data.groupby('Timestep').size()
        threshold = spike_counts.mean() + 2 * spike_counts.std()
        burst_events = spike_counts[spike_counts > threshold]
        metrics['burst_frequency'] = len(burst_events) / len(spike_counts)
    else:
        metrics['burst_frequency'] = 0
    
    return metrics


def generate_visualization(csv_prefix="", condition_name="Standard"):
    try:
        membrane_data = pd.read_csv(f'{csv_prefix}membrane_potentials.csv')
        spike_data = pd.read_csv(f'{csv_prefix}spike_raster.csv')
        activity_data = pd.read_csv(f'{csv_prefix}activity_summary.csv')
        
        metrics = calculate_advanced_metrics(membrane_data, spike_data, activity_data)
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Neural Network Analysis: {condition_name}', fontsize=16, fontweight='bold')
        
        # Membrane Potentials
        ax1 = fig.add_subplot(gs[0, 0])
        colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(membrane_data.columns)-1)))
        for i in range(min(10, len(membrane_data.columns)-1)):
            ax1.plot(membrane_data.iloc[:, i+1], alpha=0.7, color=colors[i], linewidth=1.5)
        ax1.set_title('Membrane Potentials (Sample Neurons)', fontweight='bold')
        ax1.set_ylabel('Potential (mV)')
        ax1.set_facecolor('white')
        ax1.grid(False)
        
        # Spike Raster
        ax2 = fig.add_subplot(gs[0, 1])
        if len(spike_data) > 0:
            scatter = ax2.scatter(spike_data['Timestep'], spike_data['Neuron_ID'], 
                               s=0.5, alpha=0.6, c=spike_data['Timestep'], cmap='plasma')
            ax2.set_title('Spike Raster Plot', fontweight='bold')
            ax2.set_ylabel('Neuron ID')
            plt.colorbar(scatter, ax=ax2, label='Time')
        else:
            ax2.text(0.5, 0.5, 'No Spikes Detected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Spike Raster Plot', fontweight='bold')
        ax2.set_facecolor('white')
        ax2.grid(False)
        
        # Network Activity
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(activity_data['Timestep'], activity_data['Average_Potential'], 
                color='darkblue', linewidth=2)
        ax3.set_title('Network Activity', fontweight='bold')
        ax3.set_ylabel('Average Potential (mV)')
        ax3.set_facecolor('white')
        ax3.grid(False)
        
        # Spike Rate
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.scatter(activity_data['Timestep'], activity_data['Spike_Count'], 
                   color='red', s=4, alpha=0.8)
        ax4.set_title('Spike Rate', fontweight='bold')
        ax4.set_ylabel('Spikes per Timestep')
        ax4.set_xlabel('Timestep')
        ax4.set_facecolor('white')
        ax4.grid(False)
        
        # Frequency Analysis
        ax5 = fig.add_subplot(gs[1, 0])
        if len(activity_data) > 100:
            signal_data = activity_data['Average_Potential'].values
            freqs, psd = signal.periodogram(signal_data, fs=10.0)  # Assume 10 Hz sampling
            ax5.semilogy(freqs, psd, color='purple', linewidth=2)
            ax5.set_title('Power Spectral Density', fontweight='bold')
            ax5.set_xlabel('Frequency (Hz)')
            ax5.set_ylabel('Power')
            ax5.set_facecolor('white')
            ax5.grid(False)
        
        # Synchronization Analysis
        ax6 = fig.add_subplot(gs[1, 1])
        if len(membrane_data.columns) > 3:
            # Calculate rolling correlation between neurons
            potentials = membrane_data.iloc[:, 1:6].values  # First 5 neurons
            correlations = []
            window = 100
            for i in range(window, len(potentials)):
                corr_matrix = np.corrcoef(potentials[i-window:i].T)
                correlations.append(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            
            ax6.plot(range(window, len(potentials)), correlations, color='orange', linewidth=2)
            ax6.set_title('Network Synchronization', fontweight='bold')
            ax6.set_ylabel('Correlation Coefficient')
            ax6.set_xlabel('Timestep')
            ax6.set_facecolor('white')
            ax6.grid(False)
        
        # Membrane Potential Distribution
        ax9 = fig.add_subplot(gs[1, 2])
        all_potentials = membrane_data.iloc[:, 1:].values.flatten()
        ax9.hist(all_potentials, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax9.set_title('Membrane Potential Distribution', fontweight='bold')
        ax9.set_xlabel('Potential (mV)')
        ax9.set_ylabel('Frequency')
        ax9.set_facecolor('white')
        ax9.grid(False)
        
        # Stability Analysis
        ax11 = fig.add_subplot(gs[1, 3])
        if len(activity_data) > 50:
            # Calculate running variance as stability measure
            signal_data = activity_data['Average_Potential'].values
            window = 50
            running_var = []
            for i in range(window, len(signal_data)):
                running_var.append(np.var(signal_data[i-window:i]))
            
            ax11.plot(range(window, len(signal_data)), running_var, color='brown', linewidth=2)
            ax11.set_title('Network Stability', fontweight='bold')
            ax11.set_ylabel('Running Variance')
            ax11.set_xlabel('Timestep')
            ax11.set_facecolor('white')
            ax11.grid(False)
        
        plt.tight_layout()
        plt.show()
        return metrics
        
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print('Make sure to export CSV data first.')
        return {}
    except Exception as e:
        print(f'Error creating enhanced visualization: {e}')
        return {}


def get_clinical_interpretation(metrics, condition_name, activity_data):
    interpretation = f"CLINICAL ANALYSIS:\n{condition_name}\n\n"
    
    # Synchronization interpretation
    sync = metrics.get('synchronization', 0)
    if sync > 0.7:
        interpretation += "High synchronization\n   (possible seizure activity)\n\n"
    elif sync < 0.2:
        interpretation += "Low synchronization\n   (network fragmentation)\n\n"
    else:
        interpretation += "Normal synchronization\n\n"
    
    # Entropy interpretation
    entropy_val = metrics.get('neural_entropy', 0)
    if entropy_val < 1.5:
        interpretation += "Low complexity\n   (reduced information processing)\n\n"
    elif entropy_val > 3.0:
        interpretation += "High complexity\n   (possible chaos/noise)\n\n"
    else:
        interpretation += "Normal complexity\n\n"
    
    # Burst analysis
    burst_freq = metrics.get('burst_frequency', 0)
    if burst_freq > 0.1:
        interpretation += "Frequent bursting\n   (hyperexcitability)\n\n"
    else:
        interpretation += "Normal burst activity\n\n"
    
    # Overall spike rate
    if len(activity_data) > 0:
        total_spikes = activity_data['Spike_Count'].sum()
        total_time = len(activity_data)
        spike_rate = total_spikes / total_time if total_time > 0 else 0
        
        if spike_rate < 0.1:
            interpretation += "Hypoactivity\n   (depression-like)\n"
        elif spike_rate > 0.8:
            interpretation += "Hyperactivity\n   (mania-like)\n"
        else:
            interpretation += "Normal activity level\n"
    
    return interpretation


def run_standard_simulation():
    print("Running standard neural network simulation...")
    simulator = neuron_simulator.NeuronSimulator()
    simulator.run_standard_simulation(8000)
    simulator.export_csv_data()
    metrics = simulator.calculate_stability_metrics()
    data = simulator.get_simulation_data()
    print(f"\nSimulation Complete:")
    print(f"Total timesteps: {data.total_timesteps}")
    print(f"Total spikes: {data.total_spikes}")
    print(f"Spike rate: {data.total_spikes / data.total_timesteps:.3f} spikes/timestep")
    print(f"Coefficient of variation: {metrics.coefficient_of_variation:.3f}")
    print(f"Synchrony index: {metrics.synchrony_index:.3f}")
    print(f"Network entropy: {metrics.entropy:.3f}")
    print(f"Homeostatic deviation: {metrics.homeostatic_deviation:.1f} mV")
    print("\nGenerating enhanced visualization...")
    enhanced_metrics = generate_visualization("", "Standard Network")
    return enhanced_metrics


def run_metabolic_studies():
    print("Running comprehensive metabolic dysfunction studies...")
    simulator = neuron_simulator.NeuronSimulator()
    simulator.run_metabolic_dysfunction_studies()
    print("Metabolic studies complete!")
    print("Generating comparative visualizations...")
    conditions = ['Severe_Hypoglycemia_', 'Diabetic_Ketoacidosis_', 'Cerebral_Hypoxia_', 'Mitochondrial_Dysfunction_']
    condition_names = ['Severe Hypoglycemia', 'Diabetic Ketoacidosis', 'Cerebral Hypoxia', 'Mitochondrial Dysfunction']
    for condition_prefix, condition_name in zip(conditions, condition_names):
        try:
            print(f"Analyzing {condition_name}...")
            generate_visualization(condition_prefix, condition_name)
        except Exception as e:
            print(f"Could not generate visualization for {condition_name}: {e}")


def run_mental_health_studies():
    print("Running comprehensive mental health studies...")
    conditions = create_mental_health_conditions()
    simulator = neuron_simulator.NeuronSimulator()
    results = {}
    for condition_key, condition in conditions.items():
        print(f"\nRunning {condition.name} simulation...")
        
        if 'bipolar' in condition_key:
            timesteps = 200000  # Need longer for mood cycles
        elif condition.progressive:
            timesteps = 150000  # Longer for progressive conditions
        else:
            timesteps = 100000
        
        try:
            simulator.run_metabolic_dysfunction_simulation(condition, timesteps)
            safe_name = condition.name.replace(' ', '_').replace('(', '').replace(')', '')
            simulator.export_csv_data(f"{safe_name}_")
            metrics = simulator.calculate_stability_metrics()
            data = simulator.get_simulation_data()
            print(f"=== {condition.name} Results ===")
            print(f"Total spikes: {data.total_spikes}")
            print(f"Spike rate: {data.total_spikes / data.total_timesteps:.3f}")
            print(f"Coefficient of variation: {metrics.coefficient_of_variation:.3f}")
            print(f"Synchrony index: {metrics.synchrony_index:.3f}")
            print(f"Homeostatic deviation: {metrics.homeostatic_deviation:.1f} mV")
            enhanced_metrics = generate_visualization(f"{safe_name}_", condition.name)
            results[condition_key] = {
                'metrics': metrics,
                'data': data,
                'enhanced_metrics': enhanced_metrics,
                'condition': condition
            }
        except Exception as e:
            print(f"Error simulating {condition.name}: {e}")
    
    if results:
        generate_comparative_analysis(results)
    
    return results


def generate_comparative_analysis(results):
    print("\nGenerating comparative analysis...")
    condition_names = []
    spike_rates = []
    synchrony_values = []
    entropy_values = []
    cv_values = []
    
    for _, result in results.items():
        condition_names.append(result['condition'].name)
        data = result['data']
        metrics = result['metrics']
        spike_rate = data.total_spikes / data.total_timesteps if data.total_timesteps > 0 else 0
        spike_rates.append(spike_rate)
        synchrony_values.append(metrics.synchrony_index)
        entropy_values.append(metrics.entropy)
        cv_values.append(metrics.coefficient_of_variation)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mental Health Conditions: Comparative Neural Analysis', fontsize=16, fontweight='bold')
    
    # Spike Rate Comparison
    bars1 = axes[0,0].bar(condition_names, spike_rates, color=sns.color_palette("husl", len(condition_names)))
    axes[0,0].set_title('Spike Rate Comparison', fontweight='bold')
    axes[0,0].set_ylabel('Spikes per Timestep')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhspan(0.2, 0.4, alpha=0.2, color='green', label='Normal Range')
    axes[0,0].legend()
    
    # Synchrony Comparison
    bars2 = axes[0,1].bar(condition_names, synchrony_values, color=sns.color_palette("husl", len(condition_names)))
    axes[0,1].set_title('Network Synchrony Comparison', fontweight='bold')
    axes[0,1].set_ylabel('Synchrony Index')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].axhspan(0.3, 0.7, alpha=0.2, color='green', label='Normal Range')
    axes[0,1].legend()
    
    # Entropy Comparison
    bars3 = axes[1,0].bar(condition_names, entropy_values, color=sns.color_palette("husl", len(condition_names)))
    axes[1,0].set_title('Neural Complexity (Entropy)', fontweight='bold')
    axes[1,0].set_ylabel('Entropy')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].axhspan(2.0, 3.0, alpha=0.2, color='green', label='Normal Range')
    axes[1,0].legend()
    
    # Coefficient of Variation
    bars4 = axes[1,1].bar(condition_names, cv_values, color=sns.color_palette("husl", len(condition_names)))
    axes[1,1].set_title('Neural Variability (CV)', fontweight='bold')
    axes[1,1].set_ylabel('Coefficient of Variation')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].axhspan(0.5, 1.5, alpha=0.2, color='green', label='Normal Range')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('mental_health_comparative_analysis.png', dpi=300, bbox_inches='tight')
    print("Comparative analysis saved as: mental_health_comparative_analysis.png")
    plt.show()


def run_single_condition():
    print("Available conditions:")
    print("1. Severe Hypoglycemia")
    print("2. Diabetic Ketoacidosis") 
    print("3. Cerebral Hypoxia")
    print("4. Mitochondrial Dysfunction")
    print("5. Major Depression")
    print("6. Bipolar Disorder (Manic)")
    print("7. Bipolar Disorder (Depressive)")
    print("8. Schizophrenia")
    print("9. Anxiety Disorder")
    print("10. ADHD")
    print("11. PTSD")
    
    choice = input("Enter choice (1-11): ")
    simulator = neuron_simulator.NeuronSimulator()
    condition = None
    condition_name = ""
    
    if choice == "1":
        condition = neuron_simulator.MetabolicCondition()
        condition.name = "Severe Hypoglycemia"
        condition.glucose_level = 40.0
        condition.atp_efficiency = 0.3
        condition.ion_pump_function = 0.4
        condition.neurotransmitter_synthesis = 0.5
        condition.membrane_integrity = 0.6
        condition.oxidative_stress = 0.8
        condition.progressive = True
        condition.onset_timestep = 1000
        condition_name = "Severe_Hypoglycemia"
    elif choice == "2":
        condition = neuron_simulator.MetabolicCondition()
        condition.name = "Diabetic Ketoacidosis"
        condition.glucose_level = 300.0
        condition.atp_efficiency = 0.4
        condition.ion_pump_function = 0.5
        condition.neurotransmitter_synthesis = 0.6
        condition.membrane_integrity = 0.5
        condition.oxidative_stress = 0.9
        condition.progressive = True
        condition.onset_timestep = 500
        condition_name = "Diabetic_Ketoacidosis"
    elif choice == "3":
        condition = neuron_simulator.MetabolicCondition()
        condition.name = "Cerebral Hypoxia"
        condition.glucose_level = 90.0
        condition.atp_efficiency = 0.2
        condition.ion_pump_function = 0.3
        condition.neurotransmitter_synthesis = 0.4
        condition.membrane_integrity = 0.4
        condition.oxidative_stress = 0.95
        condition.progressive = True
        condition.onset_timestep = 200
        condition_name = "Cerebral_Hypoxia"
    elif choice == "4":
        condition = neuron_simulator.MetabolicCondition()
        condition.name = "Mitochondrial Dysfunction"
        condition.glucose_level = 100.0
        condition.atp_efficiency = 0.25
        condition.ion_pump_function = 0.35
        condition.neurotransmitter_synthesis = 0.45
        condition.membrane_integrity = 0.7
        condition.oxidative_stress = 0.85
        condition.progressive = True
        condition.onset_timestep = 1500
        condition_name = "Mitochondrial_Dysfunction"
    elif choice in ["5", "6", "7", "8", "9", "10", "11"]:
        conditions = create_mental_health_conditions()
        condition_map = {
            "5": "depression",
            "6": "bipolar_manic", 
            "7": "bipolar_depressive",
            "8": "schizophrenia",
            "9": "anxiety",
            "10": "adhd",
            "11": "ptsd"
        }
        condition_key = condition_map[choice]
        condition = conditions[condition_key]
        condition_name = condition.name.replace(' ', '_').replace('(', '').replace(')', '')
    else:
        print("Invalid choice!")
        return
    
    print(f"Running {condition.name} simulation...")
    
    timesteps = 120000 if 'bipolar' in condition_name.lower() else 100000
    
    try:
        simulator.run_metabolic_dysfunction_simulation(condition, timesteps)
        simulator.export_csv_data(f"{condition_name}_")
        metrics = simulator.calculate_stability_metrics()
        data = simulator.get_simulation_data()
        print(f"\n=== {condition.name} Results ===")
        print(f"Total timesteps: {data.total_timesteps}")
        print(f"Total spikes: {data.total_spikes}")
        print(f"Spike rate: {data.total_spikes / data.total_timesteps:.3f}")
        print(f"Coefficient of variation: {metrics.coefficient_of_variation:.3f}")
        print(f"Synchrony index: {metrics.synchrony_index:.3f}")
        print(f"Network entropy: {metrics.entropy:.3f}")
        print(f"Homeostatic deviation: {metrics.homeostatic_deviation:.1f} mV")
        enhanced_metrics = generate_visualization(f"{condition_name}_", condition.name)
        return enhanced_metrics
    except Exception as e:
        print(f"Error running simulation: {e}")
        return {}


def run_comparative_analysis():
    print("Running comparative analysis of multiple conditions...")
    print("This will simulate several conditions and generate comparisons.")
    
    # Select a subset of conditions for comparison
    conditions_to_compare = [
        ("depression", "Major Depression"),
        ("bipolar_manic", "Bipolar (Manic)"), 
        ("anxiety", "Anxiety Disorder"),
        ("schizophrenia", "Schizophrenia")
    ]
    
    simulator = neuron_simulator.NeuronSimulator()
    mental_conditions = create_mental_health_conditions()
    results = {}

    # Add normal baseline
    print("Running baseline normal simulation...")
    simulator.run_standard_simulation(8000)
    simulator.export_csv_data("Normal_")
    baseline_metrics = simulator.calculate_stability_metrics()
    baseline_data = simulator.get_simulation_data()
    results['normal'] = {
        'metrics': baseline_metrics,
        'data': baseline_data,
        'condition': type('obj', (object,), {'name': 'Normal'})()
    }
    
    for condition_key, display_name in conditions_to_compare:
        print(f"Running {display_name} simulation...")
        condition = mental_conditions[condition_key]
        timesteps = 100000
        try:
            simulator.run_metabolic_dysfunction_simulation(condition, timesteps)
            safe_name = display_name.replace(' ', '_').replace('(', '').replace(')', '')
            simulator.export_csv_data(f"{safe_name}_")
            metrics = simulator.calculate_stability_metrics()
            data = simulator.get_simulation_data()
            results[condition_key] = {
                'metrics': metrics,
                'data': data,
                'condition': condition
            }
            print(f"  Spike rate: {data.total_spikes / data.total_timesteps:.3f}")
            print(f"  Synchrony: {metrics.synchrony_index:.3f}")
        except Exception as e:
            print(f"Error with {display_name}: {e}")
    
    if len(results) > 1:
        generate_comparative_analysis(results)
    
    return results


def run_progression_study():
    print("Running long-term progression study...")
    print("This simulates progressive neurodegeneration over extended time periods.")
    
    # Create a progressive neurodegenerative condition
    condition = neuron_simulator.MetabolicCondition()
    condition.name = "Progressive Neurodegeneration"
    condition.glucose_level = 90.0
    condition.atp_efficiency = 0.8  # Starts near normal, will decline
    condition.ion_pump_function = 0.9
    condition.neurotransmitter_synthesis = 0.8
    condition.membrane_integrity = 0.9
    condition.oxidative_stress = 0.3  # Will increase over time
    condition.progressive = True
    condition.onset_timestep = 10000
    simulator = neuron_simulator.NeuronSimulator()
    
    timesteps = 100000
    print(f"Running {timesteps} timestep simulation (this may take a while)...")
    
    try:
        simulator.run_metabolic_dysfunction_simulation(condition, timesteps)
        simulator.export_csv_data("Progressive_")
        metrics = simulator.calculate_stability_metrics()
        data = simulator.get_simulation_data()
        print(f"\n=== Progressive Neurodegeneration Results ===")
        print(f"Total timesteps: {data.total_timesteps}")
        print(f"Total spikes: {data.total_spikes}")
        print(f"Final spike rate: {data.total_spikes / data.total_timesteps:.3f}")
        print(f"Final synchrony: {metrics.synchrony_index:.3f}")
        print(f"Final entropy: {metrics.entropy:.3f}")
        enhanced_metrics = generate_visualization("Progressive_", "Progressive Neurodegeneration")
        return enhanced_metrics
    except Exception as e:
        print(f"Error in progression study: {e}")
        return {}


def main():
    print("This simulator models human neurons with metabolic and mental health conditions.")
    print()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            run_standard_simulation()
        elif choice == '2':
            run_metabolic_studies()
        elif choice == '3':
            run_mental_health_studies()
        elif choice == '4':
            run_single_condition()
        elif choice == '5':
            run_comparative_analysis()
        elif choice == '6':
            run_progression_study()
        elif choice == '7':
            print("Exiting simulator. Thank you!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()