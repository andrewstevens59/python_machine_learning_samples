import execute_news_signals

hedge_lookups = [
['AUD_NZD', '101-011-9454699-007', '653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'],
['AUD_CAD', '101-011-9454699-005', '653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'],
['EUR_USD', '101-011-9454699-002', '653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'],
['NZD_CAD', '101-011-9454699-003', '653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'],
['EUR_GBP', '101-011-9454699-004', '653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'],
['GBP_USD', '101-011-9454699-006', '653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'],

['CHF_JPY', '101-011-11437099-002', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['EUR_NZD', '101-011-11437099-003', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['GBP_JPY', '101-011-11437099-004', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['AUD_CHF', '101-011-11437099-005', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['EUR_AUD', '101-011-11437099-006', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['GBP_NZD', '101-011-11437099-007', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['USD_CAD', '101-011-11437099-008', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['AUD_JPY', '101-011-11437099-009', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['EUR_CAD', '101-011-11437099-010', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['USD_CHF', '101-011-11437099-011', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['EUR_CHF', '101-011-11437099-012', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],
['AUD_USD', '101-011-11437099-013', '5ef5baee1e0926af12aa68acc27091a6-db92618643f8409a8d9d8d6e94aa9928'],

['GBP_AUD', '101-011-11437155-002', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['NZD_CHF', '101-011-11437155-003', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['CAD_CHF', '101-011-11437155-004', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['EUR_JPY', '101-011-11437155-005', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['GBP_CAD', '101-011-11437155-006', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['NZD_JPY', '101-011-11437155-007', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['CAD_JPY', '101-011-11437155-008', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['GBP_CHF', '101-011-11437155-009', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['NZD_USD', '101-011-11437155-010', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
['USD_JPY', '101-011-11437155-011', '8ef4c16d095478cf86efdf4c7b28bb7b-223ab461469c4d89f96c475eaaac6250'],
]

no_hedge_lookups = [
['AUD_NZD', '101-011-11448117-002', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['AUD_CAD', '101-011-11448117-003', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['EUR_USD', '101-011-11448117-004', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['NZD_CAD', '101-011-11448117-005', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['EUR_GBP', '101-011-11448117-006', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['GBP_USD', '101-011-11448117-007', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['CHF_JPY', '101-011-11448117-008', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['EUR_NZD', '101-011-11448117-009', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['GBP_JPY', '101-011-11448117-010', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['AUD_CHF', '101-011-11448117-011', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['EUR_AUD', '101-011-11448117-012', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['GBP_NZD', '101-011-11448117-013', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['USD_CAD', '101-011-11448117-014', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['AUD_JPY', '101-011-11448117-015', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],
['EUR_CAD', '101-011-11448117-016', '827c5f120d7fad039403aff243aafdfb-b8c81c6d46c63a77a05761b884f8efda'],

['USD_CHF', '101-011-11448292-002', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['EUR_CHF', '101-011-11448292-003', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['AUD_USD', '101-011-11448292-004', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['GBP_AUD', '101-011-11448292-005', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['NZD_CHF', '101-011-11448292-006', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['CAD_CHF', '101-011-11448292-007', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['EUR_JPY', '101-011-11448292-008', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['GBP_CAD', '101-011-11448292-009', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['NZD_JPY', '101-011-11448292-010', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['CAD_JPY', '101-011-11448292-011', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['GBP_CHF', '101-011-11448292-012', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['NZD_USD', '101-011-11448292-013', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
['USD_JPY', '101-011-11448292-014', 'f324636a9d25c84e3e30a35734c3c934-09ac5ae8ffd4a8ddef8e3e2197eb8908'],
]


# hedged
hedge_lookups_all = [
["101-011-11646238-002", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-003", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-005", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-006", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-007", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-008", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-009", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-010", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-011", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-012", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-013", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-014", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-015", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],
["101-011-11646238-016", "2a9fe04bd7b4e088ebedd2707f97a120-29c24d765d794d54cad6ed974ee45e66"],

["101-011-11646385-002", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-003", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-004", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-005", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-006", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-007", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-008", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-009", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-010", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-011", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-012", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-013", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-014", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"],
["101-011-11646385-015", "cb3d2e3d7f4ff1c4b5f8815ca3ebc013-43044fd4506d59501160af5322a5e2e6"]
]

#not hedged
no_hedge_lookups_all = [
["101-011-11646462-002", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-003", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-004", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-005", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-006", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-007", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-008", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-009", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-010", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-011", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-012", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-013", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-014", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],
["101-011-11646462-015", "4ed703b0cec655c11fb972494cd738db-fc5bb63bc1f43645d60112e855997a28"],

["101-011-11646525-002", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-003", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-004", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-005", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-006", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-007", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-008", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-009", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-010", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-011", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-012", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-013", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-014", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"],
["101-011-11646525-015", "19f131c680af053ba591e18e2d4766b7-d9e5c4201bdd1cbb65be373e76614a8c"]
]


dummy_accounts = [
["101-011-11679215-014", "27ae382a6d9ba0c69e0119a3687a2b57-839feb4e11efc998c67df9bc5f500516"],
["101-011-11679215-015", "27ae382a6d9ba0c69e0119a3687a2b57-839feb4e11efc998c67df9bc5f500516"],

["101-011-14453707-009", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-010", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-011", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-012", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-013", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-014", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-015", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-016", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-017", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-018", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-019", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],
    ["101-011-14453707-020", "a124c193ca5e9f0c9b7d8865f2c70657-07d05f6615da5ea81a4e4752e63d3ddb"],


["101-011-11679313-002", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-003", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-004", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-005", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-006", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-007", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-008", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-009", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-010", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-011", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-012", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-013", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-014", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],
["101-011-11679313-015", "578d710166c1ff3b0d1ec4f79d8c0ff9-51bb7c192180f8f28586231c20c7edfa"],



["101-011-11679362-002", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-003", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-004", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-005", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-006", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-007", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-008", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-009", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-010", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-011", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-012", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-013", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-014", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],
["101-011-11679362-015", "20f1d8e3702ee99f2f2bb9d486644636-1865b5b9c9b05094e6214f37ff9c0530"],

["101-011-11679404-002", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-003", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-004", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-005", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-006", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-007", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-008", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-009", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-010", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-011", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-012", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-013", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-014", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],
["101-011-11679404-015", "1918d7df4cc75ad7232065b7e1d2d8d2-e002c9ad3ad88035e1785a5f52060d5a"],



	["101-011-11817150-002", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
	["101-011-11817150-003", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
	["101-011-11817150-004", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
	["101-011-11817150-005", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
	["101-011-11817150-006", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
	["101-011-11817150-007", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
	["101-011-11817150-008", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],

]




hedge_lookups_all = [[execute_news_signals.currency_pairs[index], hedge_lookups_all[index][0], hedge_lookups_all[index][1]] for index in range(len(hedge_lookups_all))]

no_hedge_lookups_all = [[execute_news_signals.currency_pairs[index], no_hedge_lookups_all[index][0], no_hedge_lookups_all[index][1]] for index in range(len(no_hedge_lookups_all))]



execute_news_signals.checkIfProcessRunning('execute_demo_accounts_barrier.py', execute_news_signals.sys.argv[1])

count = 0
for select_pair in execute_news_signals.sys.argv[1].split(","):
	count += 1

	if count != 1:
		handlers = execute_news_signals.trade_logger.handlers[:]
		for handler in handlers:
			handler.close()
			execute_news_signals.trade_logger.removeHandler(handler)

	execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_" + select_pair + ".log")
	execute_news_signals.file_ext_key = ""
	execute_news_signals.account_type = "fxpractice"

	for lookup in hedge_lookups:
		if lookup[0] == select_pair:
			execute_news_signals.api_key = lookup[2]

			execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_", execute_news_signals.ModelType.barrier) #demo
			break

	handlers = execute_news_signals.trade_logger.handlers[:]
	for handler in handlers:
		handler.close()
		execute_news_signals.trade_logger.removeHandler(handler)

	execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_no_hedge_" + select_pair + ".log")
	execute_news_signals.file_ext_key = "_no_hedge"

	for lookup in no_hedge_lookups:
		if lookup[0] == select_pair:
			execute_news_signals.api_key = lookup[2]

			execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_", execute_news_signals.ModelType.barrier) #demo
			break

	handlers = execute_news_signals.trade_logger.handlers[:]
	for handler in handlers:
		handler.close()
		execute_news_signals.trade_logger.removeHandler(handler)

	execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_all_" + select_pair + ".log")
	execute_news_signals.file_ext_key = "_hedge_all"

	for lookup in hedge_lookups_all:
		if lookup[0] == select_pair:
			execute_news_signals.api_key = lookup[2]

			execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier) #demo
			break

	handlers = execute_news_signals.trade_logger.handlers[:]
	for handler in handlers:
		handler.close()
		execute_news_signals.trade_logger.removeHandler(handler)

	execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_no_hedge_all_" + select_pair + ".log")
	execute_news_signals.file_ext_key = "_no_hedge_all"

	for lookup in no_hedge_lookups_all:
		if lookup[0] == select_pair:
			execute_news_signals.api_key = lookup[2]

			execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier) #demo 
			break

handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio.log")

lookup = ["101-011-11740843-003", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]
execute_news_signals.api_key = lookup[1]
execute_news_signals.account_type = "fxpractice"

all_order_ids = []
for select_pair in execute_news_signals.sys.argv[1].split(","):

    sharpe = execute_news_signals.get_sharpe(False, True, select_pair, "", execute_news_signals.ModelType.barrier)["sharpe"]
    sharpe = max(0, sharpe)
    sharpe /= 0.1

    execute_news_signals.file_ext_key = "portfolio_single_no_hedge_"
    all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_", execute_news_signals.ModelType.barrier, strategy_weight = sharpe, is_filter_member_orders = True, is_filter_members_hedge = True) 

    sharpe = execute_news_signals.get_sharpe(False, False, select_pair, "all", execute_news_signals.ModelType.barrier)["sharpe"]
    sharpe = max(0, sharpe)
    sharpe /= 0.1
    execute_news_signals.file_ext_key = "portfolio_all_no_hedge_"
    all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier, strategy_weight = sharpe, is_filter_member_orders = True, is_filter_members_hedge = False) 

for select_pair in execute_news_signals.sys.argv[1].split(","):
    execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)

dummy_account_index = 0
for auc_barrier in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]:
	for min_trade_volatility in [25, 50, 75, 100, 125, 150, 175]:

		handlers = execute_news_signals.trade_logger.handlers[:]
		for handler in handlers:
			handler.close()
			execute_news_signals.trade_logger.removeHandler(handler)

		execute_news_signals.file_ext_key = "params_auc_" + str(auc_barrier) + "_min_volatility_" + str(min_trade_volatility) 
		execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + execute_news_signals.file_ext_key + "_trade_news_release_no_hedge_all.log")
		lookup = dummy_accounts[dummy_account_index]
		execute_news_signals.api_key = lookup[1]

		trading_params = {"auc_barrier_mult" : auc_barrier, "min_trade_volatility" : min_trade_volatility}
		for select_pair in execute_news_signals.sys.argv[1].split(","):
		   
			execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier, trading_params = trading_params, strategy_weight = 0.2) #demo
			print ("demo account index", dummy_account_index, lookup)

		dummy_account_index = dummy_account_index + 1
	
handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio_auc_0.1.log")
lookup = ["101-011-11740843-010", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]
execute_news_signals.api_key = lookup[1]
execute_news_signals.account_type = "fxpractice"


all_order_ids = []
for auc_barrier in [0.1]:
    for min_trade_volatility in [25, 50, 75, 100, 125]:

        trading_params = {"auc_barrier_mult" : auc_barrier, "min_trade_volatility" : min_trade_volatility}
        for select_pair in execute_news_signals.sys.argv[1].split(","):
            execute_news_signals.file_ext_key = "params_auc_0.1_only_no_hedge_min_volatility_" + str(min_trade_volatility) 
            all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier, trading_params = trading_params, strategy_weight = 0.5, is_filter_member_orders = True, is_filter_members_hedge = False) 

for select_pair in execute_news_signals.sys.argv[1].split(","):
    execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)

handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', "demo_account_barrier_finish_" + execute_news_signals.sys.argv[1] + ".log")
execute_news_signals.trade_logger.info('Finish: ' + str(execute_news_signals.sys.argv[1])) 

