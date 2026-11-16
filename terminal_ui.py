"""SSH 终端输入；不初始化 OpenCV 的图形后端。"""
import select
import sys


class LiveUI:
    def __init__(self, backend, headless=False, input_stream=None):
        self.backend=backend
        self.headless=headless
        self.input_stream=sys.stdin if input_stream is None else input_stream

    def waitKey(self, delay):
        if not self.headless:
            return self.backend.waitKey(delay)
        if not select.select([self.input_stream],[],[],max(0,delay)/1000)[0]:
            return -1
        command=self.input_stream.readline()
        if not command:
            return ord('q')
        command=command.strip().lower()
        return ord(command) if len(command)==1 else -1

    def __getattr__(self, name):
        if self.headless and name in ('namedWindow','setMouseCallback','imshow',
                                      'destroyWindow','destroyAllWindows'):
            return lambda *args,**kwargs: None
        return getattr(self.backend,name)


def terminal_results(watch):
    from field_map import DISPLAY
    rows=[]
    for number,name in enumerate(DISPLAY,1):
        view=(watch.views or {}).get(name)
        state=None if view is None else view.conclusion
        vote=watch.votes[name]
        confirmed=vote.locked if not vote.recheck else None
        rows.append(f'{number}号：本帧={state if state is not None else "未知"}；'
                    f'已确认={confirmed if confirmed is not None else "未确认"}')
    return rows
