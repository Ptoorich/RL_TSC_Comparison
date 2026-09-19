import sumolib

# Point this at whatever .net.xml your 6int.sumocfg references
net = sumolib.net.readNet("6int_files/6int_osm.net.xml")

tl_map = {
    "tl1": "30406195",
    "tl2": "30406197",
    "tl3": "54994558",
    "tl4": "768241766",
    "tl5": "cluster_30406198_7161921427_7161921428",
    "tl6": "cluster_54994556_7161921429",
}

lane_map = {
    "tl1": ["7521010#0_0", "622835369_0", "622835369_1", "622835369_3"],
    "tl2": ["596638461#0_0", "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3"],
    "tl3": ["380081149#2_0", "42790161#2_0", "42790161#2_1", "42790161#2_2"],
    "tl4": ["42790161#0_0", "42790161#0_1", "42790161#0_2", "7521010#1_0"],
    "tl5": ["-621271001_0", "669676574#2_0", "1190301745#0_0", "-669676574#9_0",
            "622835372_0", "622835372_1", "622835372_2"],
    "tl6": ["-621271003_0", "669676574#1_0", "623540942#0_0", "623540942#0_1",
            "623540942#0_2", "-669676574#2_0", "621271001_0"],
}

for agent, tl_id in tl_map.items():
    node = net.getNode(tl_id)
    all_incoming = set()
    for edge in node.getIncoming():
        for lane in edge.getLanes():
            all_incoming.add(lane.getID())

    monitored = set(lane_map[agent])
    missing   = all_incoming - monitored
    extra     = monitored - all_incoming  # sanity check — should be empty

    print(f"\n{agent} ({tl_id})")
    print(f"  Incoming lanes total: {len(all_incoming)}")
    print(f"  Monitored:            {len(monitored)}")
    print(f"  MISSING from lane_map: {sorted(missing)}")
    if extra:
        print(f"  ⚠ In lane_map but not incoming to this node: {sorted(extra)}")