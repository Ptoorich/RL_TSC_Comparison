import traci
traci.start(["sumo", "-c", "2int_files/2int.sumocfg"])
traci.simulationStep()
print("Traffic light IDs:", traci.trafficlight.getIDList())
print("Lane IDs:", traci.lane.getIDList())
traci.close()