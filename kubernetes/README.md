# megatron-lm job runner

GKEのCredentialsを取得する
```
gcloud container clusters get-credentials xxxx --region asia-southeast1 --project xxx
```

Generate the specified number of job yaml
```
make generate NUM_NODES=4
```

名前空間をスイッチする
```
kubens default
```

Kubernetes へデプロイする
```
make apply
```

Kubernetes から削除する
```
make delete
```