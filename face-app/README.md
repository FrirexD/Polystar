# Manual downloads for project setup

## Follow the steps in order for the app to be manually setup correctly

1) ### Download all external files not cloned from the repository (list below)

- The file **inswapper_128.onnx** shall be placed in the **/face-app/models/** folder.

    Download link :
    https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing

- Put images from the celebA dataset in the project folder under the name **/data/img_align_celeba** (you may take less than all of the 200 000 files if it takes too much time)

    Download link :
    https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ

---

_Note:_ **ALL** other **models** and **libraries** are automatically fetched and downloaded from other repoitories and can take some time to complete.

---

# Docker setup for faster results and execution

2) ### Run the preprocessing docker-compose file with the command

```bash
docker-compose -f docker-compose-preprocess.yml up --build
```

#### **DO NOT MODIFY OR DELETE ANY FILE AFTER RUNNING THIS COMMAND**

3) ### After it returns 0, run the command

```bash
docker-compose -f docker-compose-preprocess.yml down
```

4) ### Run the app and wait for execution (process might take a while depending on the number of images copied in the preprocessing step).

```bash
docker-compose -f docker-compose-app.yml up --build
```

---
_Note_ : Results of best match should appear as **/data/output/best_match.jpg**
