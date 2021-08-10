import execute_news_signals
import pickle


hedge_lookups = [
#andstv916
["101-011-13186567-002", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-003", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-004", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-005", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-006", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-007", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-008", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-009", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-010", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-011", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-012", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-013", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-014", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],
["101-011-13186567-015", "4e104f5db7f0a9b979d912c85cfe197c-8a1f99be6d139f102f6ca7d3131b263d"],

#andstv917
["101-011-13186658-002", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-003", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-004", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-005", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-006", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-007", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-008", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-009", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-010", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-011", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-012", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-013", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-014", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"],
["101-011-13186658-015", "3369db6c1fbe5516ecb5e4478adffe76-a6d3f4d2f2a6cdde976d337756b187dd"]
]

no_hedge_lookups = [ 
#andstv914
["101-011-13185923-002", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-003", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-004", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-005", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-006", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-007", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-008", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-009", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-010", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-011", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-012", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-013", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-014", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],
["101-011-13185923-015", "94908a3aa0fd766a6e6cb4c75ba45ea4-78c30a58efbf7b44ec82c540902af1e7"],

#andstv915
["101-011-13186028-002", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-003", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-004", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-005", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-006", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-007", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-008", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-009", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-010", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-011", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-012", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-013", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-014", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"],
["101-011-13186028-015", "c1fe0b03b8d97c4de80f497a4646b7c0-04d1ef7487b51b31c5739bcc7d23cc1b"]
]


# hedged
hedge_lookups_all = [ 
#andstv918
["101-011-13186763-002", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-003", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-004", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-005", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-006", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-007", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-008", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-009", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-010", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-011", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-012", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-013", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-014", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],
["101-011-13186763-015", "1ea2a058b7fe42ba500dedd8e1fa169f-b7e299a0a7bf23d725f8ecbd07b0d206"],

#andstv919
["101-011-13186840-002", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-003", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-004", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-005", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-006", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-007", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-008", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-009", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-010", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-011", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-012", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-013", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-014", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"],
["101-011-13186840-015", "f48bf307ae4a2d59a17fc01e8b6cd22d-cd455f1180afb0edc8b311850d8a00b2"]
]

#not hedged
no_hedge_lookups_all = [ 
#andstv920
["101-011-13192419-002", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-003", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-004", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-005", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-006", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-007", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-008", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-009", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-010", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-011", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-012", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-013", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-014", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],
["101-011-13192419-015", "176f85e942ffe8cbe4c648afa450f33a-7978d916728b1c884eb1cfe7e6bb7ad3"],

#andstv21
["101-011-13192468-002", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-003", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-004", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-005", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-006", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-007", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-008", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-009", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-010", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-011", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-012", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-013", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-014", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"],
["101-011-13192468-015", "9aa496b4dfacc01981f78991f66294cf-57c46ce22e229856dfcd055a10d23b5a"]
]


dummy_accounts = [
    ["101-011-11817150-009", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
    ["101-011-11817150-010", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
    ["101-011-11817150-011", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
    ["101-011-11817150-012", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
    ["101-011-11817150-013", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
    ["101-011-11817150-014", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],
    ["101-011-11817150-015", "76aac83cdb6d70ce9a7bb6ae2a9928a8-159233dcb87bc2d7aaaecfd8f60bcb3d"],

    ["101-011-11817227-002", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-003", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-004", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-005", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-006", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-007", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-008", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-009", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-010", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-011", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-012", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-013", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-014", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],
    ["101-011-11817227-015", "4334d023ef8f4fc14afe6f4516ed0b43-3bad35e6bb438a7b3e2ddd83291fbed6"],

    ["101-011-11924635-003", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-004", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-005", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-006", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-007", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-008", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-009", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-010", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-011", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-012", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-013", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-014", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-015", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],
    ["101-011-11924635-016", "d094abcf9e3dd8aa92675c50509bbc28-dd7fc7e2048de2a0a9dc295184dea2f0"],

    ["101-011-11924838-002", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-003", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-004", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-005", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-006", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-007", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-008", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-009", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-010", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-011", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-012", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-013", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-014", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],
    ["101-011-11924838-015", "81a2f0b1607a4193576c3cc7b7f98504-3c5b26de7526a463af22edf17ea61edd"],

    ["101-011-11945726-002", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-003", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-004", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-005", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-006", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-007", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-008", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-009", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-010", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-011", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-012", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-013", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-014", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],
    ["101-011-11945726-015", "d36e4a4f02ca34bc5d781bf5f1263cdc-70159fe4b96d7c47c97a69aae50969b8"],

    ["101-011-14527480-001", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-002", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-003", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-004", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-005", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-006", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-007", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-008", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-009", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-010", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-011", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-012", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-013", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-014", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-015", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-016", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-017", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-018", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-019", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],
    ["101-011-14527480-020", "dd55c53b7cf20768bc45115af5dd92d5-3946a5ec6030fdd13d02298351064be5"],

]

hedge_lookups = [[execute_news_signals.currency_pairs[index], hedge_lookups[index][0], hedge_lookups[index][1]] for index in range(len(hedge_lookups))]

no_hedge_lookups = [[execute_news_signals.currency_pairs[index], no_hedge_lookups[index][0], no_hedge_lookups[index][1]] for index in range(len(no_hedge_lookups))]


hedge_lookups_all = [[execute_news_signals.currency_pairs[index], hedge_lookups_all[index][0], hedge_lookups_all[index][1]] for index in range(len(hedge_lookups_all))]

no_hedge_lookups_all = [[execute_news_signals.currency_pairs[index], no_hedge_lookups_all[index][0], no_hedge_lookups_all[index][1]] for index in range(len(no_hedge_lookups_all))]



execute_news_signals.checkIfProcessRunning('execute_demo_accounts_news_momentum.py', execute_news_signals.sys.argv[1])

'''
count = 0
for select_pair in execute_news_signals.sys.argv[1].split(","):
    count += 1

    if count != 1:
        handlers = execute_news_signals.trade_logger.handlers[:]
        for handler in handlers:
            handler.close()
            execute_news_signals.trade_logger.removeHandler(handler)

    execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_" + select_pair + ".log")
    execute_news_signals.file_ext_key = "_hedge"
    execute_news_signals.account_type = "fxpractice"

    for lookup in hedge_lookups:
    	if lookup[0] == select_pair:
    		execute_news_signals.api_key = lookup[2]

    		execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_momentum_", execute_news_signals.ModelType.time_classification) #demo
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

            execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_momentum_", execute_news_signals.ModelType.time_classification) #demo
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

            execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_momentum_all_", execute_news_signals.ModelType.time_classification) #demo
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

            execute_news_signals.process_pending_trades([lookup[1]], execute_news_signals.avg_spreads, select_pair, "/root/news_momentum_all_", execute_news_signals.ModelType.time_classification) #demo 
            break
'''

count = 0
dummy_account_index = 0
for prefix_file in ["/root/news_momentum_", "/root/news_momentum_all_"]:
    for is_normalize_signal in [0]:
        for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
            for min_trade_volatility in [25, 50, 75, 100, 125, 150]:
                count += 1

                if count != 1:
                    handlers = execute_news_signals.trade_logger.handlers[:]
                    for handler in handlers:
                        handler.close()
                        execute_news_signals.trade_logger.removeHandler(handler)

                execute_news_signals.file_ext_key = "params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) + "_" + str(prefix_file[6:]) 
                execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + execute_news_signals.file_ext_key + "_trade_news_release_no_hedge_all.log")
               
                lookup = dummy_accounts[dummy_account_index]

                execute_news_signals.api_key = lookup[1]

                trading_params = {"is_normalize_signal" : is_normalize_signal, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
                for select_pair in execute_news_signals.sys.argv[1].split(","):
                    execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, prefix_file, execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = 0.2) #demo
                    print ("demo account index", dummy_account_index, lookup)

                dummy_account_index = dummy_account_index + 1

handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio.log")

lookup = ["101-011-11740843-004", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]
execute_news_signals.api_key = lookup[1]
execute_news_signals.account_type = "fxpractice"

all_order_ids = []
for is_normalize_signal in [0]:
    for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
        for min_trade_volatility in [25, 50, 75, 100, 125, 150]:

            if (rmse_percentile != "70th_percentile" and rmse_percentile != "75th_percentile"):

                execute_news_signals.file_ext_key = "portfolio_params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) 

                trading_params = {"is_normalize_signal" : is_normalize_signal, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
                for select_pair in execute_news_signals.sys.argv[1].split(","):
                    all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/news_momentum_", execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = 0.2, is_filter_member_orders = True) #demo

for select_pair in execute_news_signals.sys.argv[1].split(","):
    execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)
  
handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio_all.log")

lookup = ["101-011-11740843-006", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]
execute_news_signals.api_key = lookup[1]
execute_news_signals.account_type = "fxpractice"

all_order_ids = []
for is_normalize_signal in [0]:
    for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
        for min_trade_volatility in [150]:

            execute_news_signals.file_ext_key = "portfolio_params_rmse_percentile_all_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) 

            trading_params = {"is_normalize_signal" : is_normalize_signal, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
            for select_pair in execute_news_signals.sys.argv[1].split(","):
                all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, "/root/news_momentum_all_", execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = 0.2, is_filter_member_orders = True) #demo

for select_pair in execute_news_signals.sys.argv[1].split(","):
    execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)
  
lookup = ["101-011-11740843-011", "fee5fff8cae39957c79e9cb1fe78276a-8a5ea905de304b6ab14cfb64fb7f44f9"]
execute_news_signals.api_key = lookup[1]
execute_news_signals.account_type = "fxpractice"

handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

with open(execute_news_signals.root_dir + "portfolio_weights.pickle", "rb") as f:
    final_map = pickle.load(f)

print (final_map.keys())

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_news_release_portfolio_weight.log")

all_order_ids = []
for singals_prefix in ["/root/news_momentum_all_", "/root/news_momentum_"]:
    for rmse_percentile in ["70th_percentile", "75th_percentile", "80th_percentile", "85th_percentile", "90th_percentile", "95th_percentile"]:
        for min_trade_volatility in [25, 50, 75, 100, 125, 150]:

            execute_news_signals.file_ext_key = "portfolio_weight_params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal)

            prefix_key = "" if (singals_prefix == "/root/news_momentum_") else "_all"
            key = "params_rmse_percentile_" + str(rmse_percentile) + "_min_volatility_" + str(min_trade_volatility)  + "_is_normalize_signal_" + str(is_normalize_signal) + "_news_momentum" + prefix_key + "__"

            if execute_news_signals.root_dir + key + "trade_news_release_no_hedge_all.log" not in final_map:
                continue

            portfolio_weight = final_map[execute_news_signals.root_dir + key + "trade_news_release_no_hedge_all.log"]
            is_new_trade = portfolio_weight > 0.01

            print (portfolio_weight, execute_news_signals.root_dir + key + "_trade_news_release_no_hedge_all.log")

            trading_params = {"is_normalize_signal" : 0, "auc_barrier_mult" : rmse_percentile, "rmse_percentile" : rmse_percentile, "min_trade_volatility" : min_trade_volatility}
            for select_pair in execute_news_signals.sys.argv[1].split(","):
                all_order_ids += execute_news_signals.process_pending_trades([lookup[0]], execute_news_signals.avg_spreads, select_pair, singals_prefix, execute_news_signals.ModelType.time_regression, trading_params = trading_params, strategy_weight = portfolio_weight, is_filter_member_orders = True, is_recover = False, is_new_trade = is_new_trade) #demo

for select_pair in execute_news_signals.sys.argv[1].split(","):
    execute_news_signals.close_all_orders_not_whitelistes([lookup[0]], select_pair, all_order_ids)
  

handlers = execute_news_signals.trade_logger.handlers[:]
for handler in handlers:
    handler.close()
    execute_news_signals.trade_logger.removeHandler(handler)

execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', "demo_account_time_regression_finish_" + execute_news_signals.sys.argv[1] + ".log")
execute_news_signals.trade_logger.info('Finish: ' + str(execute_news_signals.sys.argv[1])) 
