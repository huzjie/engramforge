# 部署

- Docker：`deploy/Dockerfile`
- K8s：`deploy/k8s.yaml`
- Helm：`deploy/helm/`

```bash
docker build -f deploy/Dockerfile -t engramforge:1.0.0 .
helm install engramforge deploy/helm
```
