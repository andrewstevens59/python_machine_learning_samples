import execute_news_trends
import execute_news_signals
import pickle


#not hedged
hedge_lookups = [ 
#andstv22
["101-011-14392464-002", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-003", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-004", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-005", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-006", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-007", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-008", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-009", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-010", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-011", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-012", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-013", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-014", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],
["101-011-14392464-015", "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"],

#andstv23
["101-011-14392618-002", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-003", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-004", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-005", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-006", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-007", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-008", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-009", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-010", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-011", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-012", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-013", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-014", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
["101-011-14392618-015", "6f9b1de62ea7d68033be6923c3f09df2-e73157d9c8229fd7feb8cf58fd056256"],
]


#andstv24
hedge_lookups_revert = [ 
["101-011-14439042-002", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-003", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-004", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-005", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-006", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-007", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-008", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-009", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-010", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-011", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-012", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-013", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-014", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],
["101-011-14439042-015", "22aa236bf8cd9036d3a1edc86f4b5df3-9cb8c330df9bca70581408c37768913e"],

#andstv25
["101-011-14439140-002", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-003", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-004", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-005", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-006", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-007", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-008", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-009", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-010", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-011", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-012", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-013", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-014", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
["101-011-14439140-015", "db20073e06ce5dce0a69644a35675ce0-28e9ea8db8ded5c24272b95b2dbe4f09"],
]

#andstv26
no_hedge_lookups_revert = [ 
["101-011-14439532-002", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-003", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-004", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-005", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-006", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-007", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-008", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-009", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-010", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-011", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-012", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-013", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-014", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],
["101-011-14439532-015", "512e93b89b33b480305f155f54c1de51-87a35a86c2a01a5447a1b2bd7f5ea603"],

#andstv27
["101-011-14439588-002", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-003", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-004", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-005", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-006", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-007", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-008", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-009", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-010", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-011", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-012", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-013", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-014", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
["101-011-14439588-015", "70d5e6973ae66d9c661f5c0fba12f7b1-778550d30cf50d9d6b6df85791ef216b"],
]


#andstv28
dummy_accounts = [
    ["101-011-11945810-002", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-003", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-004", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-005", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-006", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-007", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-008", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-009", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-010", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-011", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-012", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-013", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-014", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],
    ["101-011-11945810-015", "9267162211a18a8ab6bc9b322569eaf0-59c78cad6c29767f14a878e6d86909a4"],

    ["101-011-14453042-001", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-002", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-003", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-004", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-005", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-006", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-007", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-008", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-009", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-010", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-011", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-012", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-013", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-014", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-015", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-016", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-017", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-018", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-019", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],
    ["101-011-14453042-020", "9dae50d35ff72d021b921b84f5d968bc-d037a3bca34cfe9e2c038ae4bd28416a"],

    ["101-011-14453707-001", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-002", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-003", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-004", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-005", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-006", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-007", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-008", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],

    #andstv31

    ["101-011-14579035-001", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-002", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-003", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-004", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-005", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-006", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-007", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-008", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-009", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-010", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-011", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-012", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-013", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-014", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-015", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-016", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-017", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-018", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-019", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],
    ["101-011-14579035-020", "a3327f2ae91edc12614c919c5fb0d945-b52577d501455cd09c1cd956907b7da9"],

    #andstv32
    ["101-011-14579539-001", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-002", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-003", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-004", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-005", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-006", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-007", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-008", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-009", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-010", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-011", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-012", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-013", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-014", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-015", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-016", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-017", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-018", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-019", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
    ["101-011-14579539-020", "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"],
]


hedge_lookups = [[execute_news_trends.currency_pairs[index], hedge_lookups[index][0], hedge_lookups[index][1]] for index in range(len(hedge_lookups))]

hedge_lookups_revert = [[execute_news_trends.currency_pairs[index], hedge_lookups_revert[index][0], hedge_lookups_revert[index][1]] for index in range(len(hedge_lookups_revert))]

no_hedge_lookups_revert = [[execute_news_trends.currency_pairs[index], no_hedge_lookups_revert[index][0], no_hedge_lookups_revert[index][1]] for index in range(len(no_hedge_lookups_revert))]


execute_news_trends.checkIfProcessRunning('execute_demo_accounts_time_regression.py', execute_news_trends.sys.argv[1])

count = 0
dummy_account_index = 0
for is_normalize_signal in [0]:
    for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
        for min_trade_volatility in [25, 50, 75, 100, 125, 150]:
            count += 1

            if count != 1:
                handlers = execute_news_signals.trade_logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    execute_news_signals.trade_logger.removeHandler(handler)

            execute_news_signals.file_ext_key = "params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) 
            execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + execute_news_signals.file_ext_key + "_trade_news_release_no_hedge_all.log")
           
            lookup = dummy_accounts[dummy_account_index]

            execute_news_signals.account_type = "fxpractice"
            execute_news_signals.api_key = lookup[1]


            trading_params = {"is_normalize_signal" : is_normalize_signal, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
            for select_pair in execute_news_signals.sys.argv[1].split(","):
                execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/time_regression_", execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = 0.2) #demo
                print ("demo account index", dummy_account_index, lookup)

            dummy_account_index = dummy_account_index + 1


lookups = [
["101-011-11740843-002", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"],
["101-011-11740843-007", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"],
["101-011-11740843-008", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"],
["101-011-11740843-009", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]
]


with open(execute_news_signals.root_dir + "portfolio_weights.pickle", "rb") as f:
    final_map = pickle.load(f)


print (final_map.keys())

demo_index = 0
for is_normalize_signal in [0, 2]:

    lookup = lookups[demo_index]
    demo_index += 1 

    all_order_ids = []
    execute_news_signals.api_key = lookup[1]
    execute_news_signals.account_type = "fxpractice"

    handlers = execute_news_signals.trade_logger.handlers[:]
    for handler in handlers:
        handler.close()
        execute_news_signals.trade_logger.removeHandler(handler)

    execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio_norm_" + str(is_normalize_signal) + ".log")

    for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
        for min_trade_volatility in [25, 50, 75, 100, 125, 150]:

            execute_news_signals.file_ext_key = "portfolio_params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal)

            key = "params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) 
            if execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log" not in final_map:
                continue

            portfolio_weight = final_map[execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log"]
            is_new_trade = portfolio_weight > 0.02

            print (portfolio_weight, execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log")

            trading_params = {"is_normalize_signal" : is_normalize_signal, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
            for select_pair in execute_news_signals.sys.argv[1].split(","):
                all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/time_regression_", execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = portfolio_weight, is_filter_member_orders = True, is_recover = False, is_new_trade = is_new_trade) #demo

    for select_pair in execute_news_signals.sys.argv[1].split(","):
        execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)

with open(execute_news_signals.root_dir + "portfolio_sharpes.pickle", "rb") as f:
    final_map = pickle.load(f)

for is_normalize_signal in [0]:

    lookup = ["101-011-11740843-012", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]

    all_order_ids = []
    execute_news_signals.api_key = lookup[1]
    execute_news_signals.account_type = "fxpractice"

    handlers = execute_news_signals.trade_logger.handlers[:]
    for handler in handlers:
        handler.close()
        execute_news_signals.trade_logger.removeHandler(handler)

    execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio_sharpes_norm_" + str(is_normalize_signal) + ".log")

    for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
        for min_trade_volatility in [25, 50, 75, 100, 125, 150]:

            execute_news_signals.file_ext_key = "portfolio_sharpes_params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal)

            key = "params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) 
            if execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log" not in final_map:
                continue

            portfolio_weight = final_map[execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log"]
            is_new_trade = portfolio_weight > 0.02

            print (portfolio_weight, execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log")

            trading_params = {"is_normalize_signal" : is_normalize_signal, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
            for select_pair in execute_news_signals.sys.argv[1].split(","):
                all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/time_regression_", execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = portfolio_weight, is_filter_member_orders = True, is_recover = False, is_new_trade = is_new_trade) #demo

    for select_pair in execute_news_signals.sys.argv[1].split(","):
        execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)
     

handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', "demo_account_time_regression_finish_" + execute_news_signals.sys.argv[1] + ".log")
execute_news_signals.trade_logger.info('Finish: ' + str(execute_news_signals.sys.argv[1])) 
