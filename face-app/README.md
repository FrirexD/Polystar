# Setup for AI models

All models should be automatically downloaded from the **docker-compose** command except from **ONE FILE**

Once the repo is cloned, the file **inswapper_128.onnx** shall be manually downloaded and placed in the **/face-app/models/** folder.

**inswapper_128.onnx** Download link :
[https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing)

Then, you might run the docker-compose preprocessing process before testing the app.

_Note:_ **ALL** other **models** files are automatically fetched and downloaded from the InsightFace repo and can take some time to complete.

---

# Docker setup for faster results and execution

1) Run the preprocessing docker-compose file with the command

```bash
docker-compose -f docker-compose-preprocess.yml up --build
```

**DO NOT MODIFY OR DELETE ANY FILE**

2) Then terminate the preprocessing app after its return 0 state with the command

```bash
docker-compose -f docker-compose-preprocess.yml down
```

3) Finally, run the app and wait for execution (process might take a while depending on the number of images copied in the preprocessing step).

```bash
docker-compose -f docker-compose-app.yml up --build
```

Result of best match should appear in **/data/output/best_match.jpg**
