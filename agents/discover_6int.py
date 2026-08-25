import traci

traci.start(["sumo", "-c", "6int_files/6int.sumocfg"])
traci.simulationStep()

tl_ids = traci.trafficlight.getIDList()
print("Traffic light IDs:", tl_ids)

for tl in tl_ids:
    print(f"\nTraffic light: {tl}")
    controlled = traci.trafficlight.getControlledLanes(tl)
    approach_lanes = [l for l in controlled if not l.startswith(':')]
    seen = set()
    unique = []
    for lane in approach_lanes:
        if lane not in seen:
            seen.add(lane)
            unique.append(lane)
    print(f"  Approach lanes: {unique}")
    logic = traci.trafficlight.getAllProgramLogics(tl)[0]
    print(f"  Number of phases: {len(logic.phases)}")
    for i, phase in enumerate(logic.phases):
        print(f"    Phase {i}: duration={phase.duration}s  state={phase.state}")

traci.close()