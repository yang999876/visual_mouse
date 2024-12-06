import threading

class EventSystem:
    def __init__(self):
        self.listeners = {}
        self.lock = threading.Lock()  # 用于确保线程安全

    def subscribe(self, event_type, listener):
        """订阅事件"""
        with self.lock:
            if event_type not in self.listeners:
                self.listeners[event_type] = []
            self.listeners[event_type].append(listener)

    def unsubscribe(self, event_type, listener):
        """取消订阅事件"""
        with self.lock:
            if event_type in self.listeners:
                self.listeners[event_type].remove(listener)

    def publish(self, event_type, *args):
        """发送事件"""
        with self.lock:
            if event_type in self.listeners:
                for listener in self.listeners[event_type]:
                    listener(*args)

    def EXIT(self):
        self.listeners = {}

event_system = EventSystem()

if __name__ == "__main__":
    class A:
        def __init__(self):
            event_system.subscribe("a", self.on_event)

        def on_event(self, data):
            print(self, data)

    a = A()

    def listener_function(data):
        print(f"Received data: {data}")

    event_system.publish('a', {'message': 'Hello world!'})
    event_system.publish('b', {'message': 'Goodbye world!'})
