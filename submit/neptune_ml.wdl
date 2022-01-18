version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task train {
    input {
        File MAIN_PY
        File ML_CONFIG
        File ckpt_previous_run
        File credentials_json
        String git_repo
        String git_branch_or_commit
        Int cpus_count
        Int gpus_count
        String gpus_type
    }


    command <<<

        exec_dir=$(pwd)
        echo "--> $exec_dir"
        echo "START --> Content of exectution dir"
        echo $(ls)

        # 2. clone the repository in the checkout_dir
        # for public repository use:
        # git clone ~{git_repo} ./checkout_dir
        # for private repository use:
        github_token=$(cat ~{credentials_json} | grep -o '"GITHUB_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$' | sed 's/"//g')
        git_repo_with_token=$(echo ~{git_repo} | sed "s/github/$github_token@github/")
        git clone $git_repo_with_token ./checkout_dir

        # 3. checkout the branch
        cd ./checkout_dir
        git checkout ~{git_branch_or_commit}
        echo "AFTER GIT --> Content of checkout dir"
        echo $(ls)
        
        # 4. Install the package
        #    and create links from delocalized files and give them the name you expects
        pip install .  # this means that you package must have a setup.py file
        ln -s ~{MAIN_PY} ./main.py  
        ln -s ~{ML_CONFIG} ./config.yaml
        ln -s $exec_dir/my_checkpoint.ckpt ./preemption_ckpt.pt  # this is to resume a pre-empted run     (it has precedence)
        ln -s ~{ckpt_previous_run} ./old_run_ckpt.pt             # this is to resume from a previous run  (secondary)

        echo "AFTER CHANGING NAMES --> Content of checkout dir"
        echo $(ls)

        # Install missing packages not already included in the docker image (if any)
        # pip install xxxx
        pip install google-cloud-storage
        pip install colorcet        
 
        # 5. run python code only if NEPTUNE credentials are found
        neptune_token=$(cat ~{credentials_json} | grep -o '"NEPTUNE_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$')
        if [ ! -z $neptune_token ]; then
           export NEPTUNE_API_TOKEN=$neptune_token
           python main.py --from_yaml ./config.yaml
        fi

        echo "ABOUT TO QUIT JOB"
        echo "I am in $(pwd)"
        echo $(ls)
    >>>
    
#    runtime {
#          docker: "python"
#          cpu: 1
#          preemptible: 3
#    }
    
    runtime {
         docker: "us.gcr.io/broad-dsde-methods/tissue_purifier:0.0.3"
         bootDiskSizeGb: 200
         memory: "26G"
         cpu: cpus_count
         zones: "us-east1-d us-east1-c"
         gpuCount: gpus_count
         gpuType: gpus_type
         maxRetries: 0
         preemptible: 0
         checkpointFile: "my_checkpoint.ckpt"
    }

}

workflow neptune_ml {

    input {
        File MAIN_PY
        File ML_CONFIG 
        File ckpt_previous_run
        File credentials_json
        String git_repo
        String git_branch_or_commit 
        Int cpus_count
        Int gpus_count
        String gpus_type
    }

    call train { 
        input :
            MAIN_PY = MAIN_PY,
            ML_CONFIG = ML_CONFIG,
            credentials_json = credentials_json,
            ckpt_previous_run = ckpt_previous_run,
            git_repo = git_repo,
            git_branch_or_commit = git_branch_or_commit,
            cpus_count = cpus_count,
            gpus_count = gpus_count,
            gpus_type = gpus_type
    }
}
