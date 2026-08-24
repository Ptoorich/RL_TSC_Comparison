import traci

traci.start(["sumo", "-c", "2int_files/2int.sumocfg"])
traci.simulationStep()

tl_ids = ['30406197', 'cluster_30406198_7161921427_7161921428']

for tl in tl_ids:
    print(f"\nTraffic light: {tl}")
    logic = traci.trafficlight.getAllProgramLogics(tl)[0]
    print(f"  Number of phases: {len(logic.phases)}")
    for i, phase in enumerate(logic.phases):
        print(f"  Phase {i}: duration={phase.duration}s  state={phase.state}")

traci.close()
