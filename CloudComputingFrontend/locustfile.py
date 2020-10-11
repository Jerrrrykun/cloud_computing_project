from locust import HttpUser, TaskSet, between, task


class UserBehavior(TaskSet):
    @task(2)
    def index(self):
        self.client.get("/")
    
    @task(1)
    def service(self):
        self.client.get("/nlp")
        self.client.get("/cv")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(5.0, 9.0)