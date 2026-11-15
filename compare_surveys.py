"""比较相同场地/起点的多轮实测：同时减少路程和观测次数。"""
import argparse
import json
from pathlib import Path


def frontier(rows):
    """只淘汰两项均不优、且至少一项更差的方案；不隐含任意权重。"""
    return [row for row in rows if not any(
        other['distance_m'] <= row['distance_m']
        and other['observation_count'] <= row['observation_count']
        and (other['distance_m'] < row['distance_m']
             or other['observation_count'] < row['observation_count'])
        for other in rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.reports or sorted((Path(__file__).parent/'debug_captures').glob('survey_*.json'))
    rows = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            metrics = report.get('metrics')
            if not report.get('complete') or metrics is None:
                continue
            rows.append(dict(report=str(path), **metrics))
        except (OSError, ValueError) as error:
            print(f'跳过 {path}: {error}')
    if not rows:
        print('暂无带路程和观测次数的完整实测记录。')
        return
    best = frontier(rows)
    print('仅比较相同场地、起点和物体布置的轮次；★为两项指标未被其他轮次同时超过的候选。')
    for row in sorted(rows, key=lambda r: (r['observation_count'], r['distance_m'])):
        print(f"{'★' if row in best else ' '} {row['observation_count']}次  {row['distance_m']:.3f}米  {row['report']}")
    print('路程为有效位姿折线累计值，含定位噪声；候选是已测轮次中的较优解。')


if __name__ == '__main__':
    main()
