# Setup for AI models

All models should be automatically downloaded from the **docker-compose** command except from **ONE FILE**

Once the repo is cloned, executing this command should create a folder in **face-app** called **models** which stores models for the AI
```bash
docker-compose up --build
```

The file **inswapper_128.onnx** shall be manually downloaded and placed in the **models** folder.

**inswapper_128.onnx** Download link :
https://www.reddit.com/r/midjourney/comments/13pnraj/please_reupload_inswapper_128onnx/?rdt=59740

<br>
**ALL** other **models** files are automatically fetched and downloaded from the InsightFace repo and can take some time to complete.