import json
from draft import api


def collect(api, number_matches, last_match_id=None):
    matches = []
    while len(matches) < number_matches:
        new_matches = api.public_matches(last_match_id)
        if not new_matches:
            break
        last_match_id = new_matches[-1]['match_id']
        matches.extend(new_matches)
        print(len(matches))
    return matches


def filter(matches, min_rank):
    return [match for match in matches if match['avg_rank_tier'] >= min_rank]


def save(matches, file):
    with open(file, 'w') as f:
        for match in matches:
            f.write(json.dumps(match) + '\n')


def run():
    a = api.Api('45b32bd3-0386-4ee7-ac48-da17ff557d3f')
    matches = collect(a, 1e6)
    save(matches, 'new-data.json')
    ancient_matches = filter(matches, 60)
    save(ancient_matches, 'new-data-ancient.json')


if __name__ == '__main__':
    run()
