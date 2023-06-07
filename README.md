# Deploying Miniature GPT on Flask, Docker, Kubernetes

![Screenshot 2023-06-06 at 11 51 13 PM](https://github.com/chrislevn/RestAPI-MLOps/assets/32094007/24349e12-a19a-4c09-95a6-ba5305a89204)

## What it does
Deploy a Machine Learning model (Miniature GPT with movie reviews dataset) with Docker, Kubernetes, Flask API. 

## How to run: 
1. Create a hub on Docker - https://hub.docker.com/ (As of current version, Kubernetes will take public image. ) 
2. Login to docker `docker login`
3. Build Docker image (current port is 5002). \
`docker build -t tagname`\
`docker push image:tagname`
3. Rename image (line 32) with the image name from your Docker Hub. 
4. Run kubernetes
`kubectl create -f deployment.yaml` 
`kubectl apply -f deployment.yaml` (this will trigger deployment and load balancer)
5. Run minikube
`minikube start`
`minikube service list` (get the list of current service. get the service running your name)
`minikube service <SERVICE_NAME>`
6. Test the API with Postman

## Future developments
- Connect Kubernetes with private Docker image
- Add CI/CD 
- Add monitoring 
