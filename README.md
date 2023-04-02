### What is this code
This is a prototype of a pipleline for processing of .tmx files. 

The project reads .tmx files and puts them into Kafka (into unprocessed_topic), then takes them from Kafka and does some basic cleaning, puts them into Kafka (into filtered_topic). Then the cleaned pairs are read from Kafka and saved to the disk locally.

After that, the check of missalignment can be done. It relies on LASER model to do so. We have a LASER endpoint running (docker/laser_app), where we send sentence pairs to encode them, after that we perform a check based on cosine similarity. If the difference is below a chosen threshold, we skip the sentence pair.

### How to run this code
To start Kafka and LASER server, in the docker directory, run:

> docker-compose up

In a virtual env:

Install dependencies
> python3 -m pip install -r requirements.txt

Parse tmx files from the given input and put them into Kafka
> python3 tmx_proc/cli.py parse-tmx "resources/*.tmx"

Consume from Kafka, clean and put "back"
> python3 tmx_proc/cli.py clean-pairs

Read cleaned pairs from Kafka and save on disk to the given output
> python3 tmx_proc/cli.py write-to-file results/

Filter misaligned pairs from the input file(s) and save to the given output
> python3 tmx_proc/cli.py filter-misalignment "results/*" results_filtered/

### Scalability proposal

To scale the whole prototype, we could, for example, do the following:

1. Skip the saving step after cleaning and put the cleaned pairs to another Kafka topic (or do both if necessary)
2. Make a similar worker/consumer for filtering misaligned pairs so we could run multiple processes in parallel for this task as well
3. Have multiple endpoints to embed the sentences using LASER
4. Run it all on a cluster, orchestrated by Kebernetes or a similar solution 
5. Move data storage from local to cloud (e.g. s3) as fsspec is file-system-agnostic and allows reading from distributed data storage
