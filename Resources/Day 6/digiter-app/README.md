# Digiter app 

```bash
docker-compose up -d --build
```

for building docker images

```bash
docker-compose down
```
for shutting down images gracefully

```bash
docker-compose up -d
```

for loading images 

```bash
docker-compose exec dev python /app/digiter/models/nn_model/train.py 
```
for training the model
