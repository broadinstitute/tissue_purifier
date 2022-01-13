# The image corresponding to this dosckerfile can be obtained by 
#docker pull us.gcr.io/broad-dsde-methods/tissue_purifier:0.0.3

FROM pytorchlightning/pytorch_lightning:latest-py3.9-torch1.9

RUN yes | pip3 install -U psutil 
RUN yes | pip3 install -U matplotlib  
RUN yes | pip3 install -U scikit-image 
RUN yes | pip3 install -U neptune-client 
RUN yes | pip3 install -U python-igraph  
RUN yes | pip3 install -U leidenalg 
RUN yes | pip3 install -U Pillow  
RUN yes | pip3 install -U pyro-ppl 
RUN yes | pip3 install -U ipython 
RUN yes | pip3 install -U pyyaml
RUN yes | pip3 install -U scikit-learn
RUN yes | pip3 install -U umap-learn
RUN yes | pip3 install -U torchmetrics
RUN yes | pip3 install -U pandas
RUN yes | pip3 install -U google-cloud-storage
RUN yes | pip3 install -U scanpy

