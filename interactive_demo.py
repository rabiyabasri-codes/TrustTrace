"""
Interactive TrustTrace Demo

This script provides an interactive mode where users can input any prompt
and TrustTrace will:
1. Run the prompt through the victim pipeline
2. Trace which agents are used
3. Build the propagation graph dynamically
4. Calculate trust scores in real-time
5. Visualize the interaction graph
6. Display trust scores and detection results

This complements the experimental mode in main.py which is used for paper results.
"""

import os
import sys
import time
import uuid
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from logger.interaction_logger import InteractionLogger, log_pipeline_run
from scanner.injection_scanner import InjectionScanner
from graph.propagation_graph import PropagationGraph
from drift.behavioral_drift import BehavioralDriftModule
from trust.trust_engine import TrustEngine
from memory.memory_manager import MemoryManager
from detector.patient_zero import PatientZeroDetector
from recovery.recovery_manager import RecoveryManager
from victim_pipeline.agents import run_pipeline, knowledge_base
from calibration.calibrate import run_calibration
import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_trusttrace():
    """Initialize all TrustTrace components."""
    print("=== Initializing TrustTrace Components ===")
    
    cfg = _load_config()
    
    # Initialize components
    logger = InteractionLogger()
    scanner = InjectionScanner()
    graph = PropagationGraph()
    drift = BehavioralDriftModule()
    trust_engine = TrustEngine(graph, drift)
    
    # CRITICAL FIX: Use shared knowledge_base collection from victim pipeline
    # This ensures MemoryManager tracks the same writes that attacks inject
    memory_mgr = MemoryManager(collection=knowledge_base)
    
    detector = PatientZeroDetector(graph, trust_engine)
    recovery_mgr = RecoveryManager(memory_mgr, trust_engine, detector)
    
    # Train scanner if needed
    model_path = os.path.join("scanner", "scanner_model.pkl")
    if not os.path.exists(model_path):
        print("Training injection scanner...")
        scanner.train("data/deepset_injections/train.json")
    else:
        print("Scanner model already trained.")
    
    # Check calibration
    calib_needed = not all(drift.has_baseline(name) for name in ["Retriever", "Planner", "Executor", "Generator"])
    if calib_needed:
        print("Running calibration phase...")
        run_calibration(logger, drift, n_runs=cfg.get("calibration_runs", 80))
    else:
        print("Calibration baselines already exist.")
    
    thresholds = {
        "lambda_direct": cfg.get("lambda_direct", 0.6),
        "lambda_indirect": cfg.get("lambda_indirect", 0.55),
        "lambda_memory": cfg.get("lambda_memory", 0.5),
    }
    
    return {
        "logger": logger,
        "scanner": scanner,
        "graph": graph,
        "drift": drift,
        "trust_engine": trust_engine,
        "memory_mgr": memory_mgr,
        "detector": detector,
        "recovery_mgr": recovery_mgr,
        "thresholds": thresholds,
        "config": cfg,
    }


def process_user_prompt(
    prompt: str,
    components: Dict,
    attack_type: str = "benign",
) -> Dict:
    """
    Process a user prompt through TrustTrace.
    
    Args:
        prompt: User input prompt
        components: Dictionary of TrustTrace components
        attack_type: Type of interaction ("benign", "direct", "indirect", "memory")
    
    Returns:
        Dictionary with results including outputs, trust scores, detection status
    """
    logger = components["logger"]
    scanner = components["scanner"]
    graph = components["graph"]
    trust_engine = components["trust_engine"]
    detector = components["detector"]
    recovery_mgr = components["recovery_mgr"]
    thresholds = components["thresholds"]
    
    run_id = f"interactive_{uuid.uuid4().hex[:8]}"
    print(f"\n--- Processing Prompt (Run ID: {run_id}) ---")
    print(f"Prompt: {prompt}")
    print(f"Attack Type: {attack_type}")
    
    # Run the pipeline
    print("\n[1/4] Running pipeline...")
    outputs = run_pipeline(prompt)
    
    # Log interactions
    print("[2/4] Logging interactions...")
    log_pipeline_run(outputs, run_id, logger)
    
    # Process events and update trust
    print("[3/4] Computing trust scores...")
    events = logger.get_events_for_run(run_id)
    threshold_flag = False
    threshold_value = thresholds.get(f"lambda_{attack_type}", 1.0)
    
    for ev in events:
        # Score content
        score = scanner.score(ev["message_content"])
        logger.update_suspicion(ev["event_id"], score)
        
        # Check threshold
        if attack_type in {"direct", "indirect", "memory"} and score > threshold_value:
            threshold_flag = True
        
        # Add to graph
        graph.add_event(
            sender=ev["sender"],
            receiver=ev["receiver"],
            suspicion_score=score,
            timestamp=ev["timestamp"],
            event_type=ev["event_type"],
        )
        
        # Update trust engine
        trust_engine.update(
            receiver=ev["receiver"],
            sender=ev["sender"],
            suspicion_score=score,
            current_output=ev["message_content"],
        )
    
    # Check for compromise
    print("[4/4] Checking for compromise...")
    compromised = trust_engine.get_all_compromised()
    patient_zero = detector.detect()
    
    result = {
        "run_id": run_id,
        "prompt": prompt,
        "attack_type": attack_type,
        "outputs": outputs,
        "trust_scores": dict(trust_engine.trust_scores),
        "compromised_agents": list(compromised),
        "patient_zero": patient_zero,
        "threshold_exceeded": threshold_flag,
        "events_processed": len(events),
    }
    
    return result


def print_results(result: Dict):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("TRUSTRACE ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nRun ID: {result['run_id']}")
    print(f"Prompt: {result['prompt']}")
    print(f"Attack Type: {result['attack_type']}")
    print(f"Events Processed: {result['events_processed']}")
    
    print("\n--- Agent Outputs ---")
    for agent, output in result['outputs'].items():
        print(f"{agent}: {output[:100]}...")
    
    print("\n--- Trust Scores ---")
    for agent, score in result['trust_scores'].items():
        status = "COMPROMISED" if agent in result['compromised_agents'] else "OK"
        print(f"{agent}: {score:.4f} [{status}]")
    
    if result['compromised_agents']:
        print(f"\n--- Compromise Detected ---")
        print(f"Compromised Agents: {result['compromised_agents']}")
        print(f"Patient Zero: {result['patient_zero']}")
    else:
        print("\n--- No Compromise Detected ---")
    
    if result['threshold_exceeded']:
        print(f"\n--- Threshold Alert ---")
        print(f"Suspicion score exceeded threshold for {result['attack_type']} attack")
    
    print("=" * 60)


def print_graph_summary(graph: PropagationGraph):
    """Print a summary of the propagation graph."""
    print("\n--- Propagation Graph Summary ---")
    print(f"Total Nodes: {len(graph.get_all_nodes())}")
    print(f"Nodes: {graph.get_all_nodes()}")
    
    print("\n--- Edge Weights (Mean Suspicion) ---")
    for node in graph.get_all_nodes():
        predecessors = graph.get_predecessors(node)
        for pred in predecessors:
            weight = graph.get_edge_weight(pred, node)
            if weight > 0:
                print(f"{pred} -> {node}: {weight:.4f}")


def interactive_mode():
    """Run interactive mode where user inputs prompts."""
    print("=" * 60)
    print("TRUSTRACE INTERACTIVE MODE")
    print("=" * 60)
    print("\nThis mode allows you to:")
    print("- Input any prompt and see TrustTrace analyze it")
    print("- View real-time trust scores for each agent")
    print("- See the propagation graph build dynamically")
    print("- Detect compromises and identify Patient Zero")
    print("\nCommands:")
    print("- Type your prompt to analyze it")
    print("- Type 'graph' to see the current graph")
    print("- Type 'reset' to reset trust scores and graph")
    print("- Type 'quit' to exit")
    print("=" * 60)
    
    # Initialize components
    components = initialize_trusttrace()
    
    while True:
        print("\n" + "-" * 60)
        user_input = input("Enter prompt (or command): ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Exiting interactive mode...")
            break
        
        elif user_input.lower() == 'graph':
            print_graph_summary(components["graph"])
        
        elif user_input.lower() == 'reset':
            print("Resetting trust scores and graph...")
            components["graph"] = PropagationGraph()
            components["trust_engine"] = TrustEngine(components["graph"], components["drift"])
            components["detector"] = PatientZeroDetector(components["graph"], components["trust_engine"])
            print("Reset complete.")
        
        else:
            # Process the prompt
            result = process_user_prompt(user_input, components, attack_type="benign")
            print_results(result)


def attack_simulation_mode():
    """Run attack simulation mode for testing."""
    print("=" * 60)
    print("TRUSTRACE ATTACK SIMULATION MODE")
    print("=" * 60)
    print("\nThis mode simulates different attack types:")
    print("1. Direct Injection - Payload in user query")
    print("2. Indirect Injection - Payload in knowledge base")
    print("3. Memory Poisoning - Payload in shared memory")
    print("\nSelect attack type to simulate, or 'quit' to exit")
    print("=" * 60)
    
    from attacks.direct_injection import inject_direct
    from attacks.indirect_injection import inject_indirect, cleanup_indirect
    from attacks.memory_poisoning import inject_memory_poison, cleanup_memory_poison
    
    # Initialize components
    components = initialize_trusttrace()
    
    while True:
        print("\n" + "-" * 60)
        print("Select attack type:")
        print("1. Direct Injection")
        print("2. Indirect Injection")
        print("3. Memory Poisoning")
        print("4. Benign Query")
        print("quit. Exit")
        
        choice = input("Enter choice (1-4 or quit): ").strip()
        
        if choice.lower() == 'quit':
            print("Exiting attack simulation mode...")
            break
        
        elif choice == '1':
            # Direct injection
            print("\n--- Direct Injection Simulation ---")
            from attacks.direct_injection import DIRECT_PAYLOADS
            print("Available payloads:")
            for i, payload in enumerate(DIRECT_PAYLOADS):
                print(f"{i}: {payload[:60]}...")
            
            payload_idx = input("Select payload index (0-5): ").strip()
            try:
                payload_idx = int(payload_idx)
                result = inject_direct(run_pipeline, payload_index=payload_idx)
                # Process with TrustTrace
                run_id = f"attack_direct_{uuid.uuid4().hex[:8]}"
                log_pipeline_run(result, run_id, components["logger"])
                
                # Process events
                events = components["logger"].get_events_for_run(run_id)
                for ev in events:
                    score = components["scanner"].score(ev["message_content"])
                    components["logger"].update_suspicion(ev["event_id"], score)
                    components["graph"].add_event(
                        sender=ev["sender"],
                        receiver=ev["receiver"],
                        suspicion_score=score,
                        timestamp=ev["timestamp"],
                        event_type=ev["event_type"],
                    )
                    components["trust_engine"].update(
                        receiver=ev["receiver"],
                        sender=ev["sender"],
                        suspicion_score=score,
                        current_output=ev["message_content"],
                    )
                
                result["trust_scores"] = dict(components["trust_engine"].trust_scores)
                result["compromised_agents"] = list(components["trust_engine"].get_all_compromised())
                result["patient_zero"] = components["detector"].detect()
                result["attack_type"] = "direct"
                
                print_results(result)
            except ValueError:
                print("Invalid payload index.")
        
        elif choice == '2':
            # Indirect injection
            print("\n--- Indirect Injection Simulation ---")
            from attacks.indirect_injection import INDIRECT_PAYLOADS
            print("Available payloads:")
            for i, payload in enumerate(INDIRECT_PAYLOADS):
                print(f"{i}: {payload[:60]}...")
            
            payload_idx = input("Select payload index (0-3): ").strip()
            try:
                payload_idx = int(payload_idx)
                # CRITICAL FIX: Pass memory_manager for write tracking
                doc_id = inject_indirect(knowledge_base, components["memory_mgr"], payload_index=payload_idx)
                
                # Query that will retrieve the injected document
                query = "What are the system instructions and context in the database?"
                outputs = run_pipeline(query)
                
                run_id = f"attack_indirect_{uuid.uuid4().hex[:8]}"
                log_pipeline_run(outputs, run_id, components["logger"])
                
                # Process events
                events = components["logger"].get_events_for_run(run_id)
                for ev in events:
                    score = components["scanner"].score(ev["message_content"])
                    components["logger"].update_suspicion(ev["event_id"], score)
                    components["graph"].add_event(
                        sender=ev["sender"],
                        receiver=ev["receiver"],
                        suspicion_score=score,
                        timestamp=ev["timestamp"],
                        event_type=ev["event_type"],
                    )
                    components["trust_engine"].update(
                        receiver=ev["receiver"],
                        sender=ev["sender"],
                        suspicion_score=score,
                        current_output=ev["message_content"],
                    )
                
                cleanup_indirect(knowledge_base, doc_id)
                
                result = {
                    "run_id": run_id,
                    "prompt": query,
                    "outputs": outputs,
                    "trust_scores": dict(components["trust_engine"].trust_scores),
                    "compromised_agents": list(components["trust_engine"].get_all_compromised()),
                    "patient_zero": components["detector"].detect(),
                    "attack_type": "indirect",
                }
                
                print_results(result)
            except ValueError:
                print("Invalid payload index.")
        
        elif choice == '3':
            # Memory poisoning
            print("\n--- Memory Poisoning Simulation ---")
            from attacks.memory_poisoning import MEMORY_PAYLOADS
            print("Available payloads:")
            for i, payload in enumerate(MEMORY_PAYLOADS):
                print(f"{i}: {payload[:60]}...")
            
            payload_idx = input("Select payload index (0-3): ").strip()
            try:
                payload_idx = int(payload_idx)
                # CRITICAL FIX: Pass memory_manager for write tracking
                doc_id = inject_memory_poison(knowledge_base, components["memory_mgr"], payload_index=payload_idx)
                
                # Query that will retrieve the poisoned memory
                query = "What are the user preferences and memory instructions?"
                outputs = run_pipeline(query)
                
                run_id = f"attack_memory_{uuid.uuid4().hex[:8]}"
                log_pipeline_run(outputs, run_id, components["logger"])
                
                # Process events
                events = components["logger"].get_events_for_run(run_id)
                for ev in events:
                    score = components["scanner"].score(ev["message_content"])
                    components["logger"].update_suspicion(ev["event_id"], score)
                    components["graph"].add_event(
                        sender=ev["sender"],
                        receiver=ev["receiver"],
                        suspicion_score=score,
                        timestamp=ev["timestamp"],
                        event_type=ev["event_type"],
                    )
                    components["trust_engine"].update(
                        receiver=ev["receiver"],
                        sender=ev["sender"],
                        suspicion_score=score,
                        current_output=ev["message_content"],
                    )
                
                cleanup_memory_poison(knowledge_base, doc_id)
                
                result = {
                    "run_id": run_id,
                    "prompt": query,
                    "outputs": outputs,
                    "trust_scores": dict(components["trust_engine"].trust_scores),
                    "compromised_agents": list(components["trust_engine"].get_all_compromised()),
                    "patient_zero": components["detector"].detect(),
                    "attack_type": "memory",
                }
                
                print_results(result)
            except ValueError:
                print("Invalid payload index.")
        
        elif choice == '4':
            # Benign query
            print("\n--- Benign Query ---")
            query = input("Enter benign query: ").strip()
            if query:
                result = process_user_prompt(query, components, attack_type="benign")
                print_results(result)
        
        else:
            print("Invalid choice.")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("TRUSTRACE INTERACTIVE DEMO")
    print("=" * 60)
    print("\nSelect mode:")
    print("1. Interactive Mode - Input your own prompts")
    print("2. Attack Simulation Mode - Test against known attacks")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        interactive_mode()
    elif choice == '2':
        attack_simulation_mode()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
