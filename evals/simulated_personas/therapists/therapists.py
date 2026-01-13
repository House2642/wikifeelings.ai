from baseline_v1 import base_app
class Therapist:
    def __init__(self, app, empty_state) -> None:
        self.app = app
        self.empty_state = empty_state
        
    def get_initial_state(self, messages):
        temp = self.empty_state
        temp["messages"] = messages
        return temp
    

baseline = Therapist(base_app, {"messages": [], "reasoning_trace" : []})
