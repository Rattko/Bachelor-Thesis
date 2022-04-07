#!/usr/bin/env zsh

export KAGGLE_USERNAME='radovanhaluska'
export KAGGLE_KEY='1787a2bbaec8d5d2f35653557dc7f409'

download_fraud_dataset() {
    # https://www.kaggle.com/mlg-ulb/creditcardfraud

    kaggle datasets download mlg-ulb/creditcardfraud \
        --path=$1 --quiet --unzip

    mv $1/creditcard.csv $1/data.csv
}

download_asteroid_dataset() {
    # https://www.kaggle.com/sakhawat18/asteroid-dataset

    kaggle datasets download sakhawat18/asteroid-dataset \
        --path=$1 --quiet --unzip

    mv $1/dataset.csv $1/data.csv
}

download_insurance_dataset() {
    # https://www.kaggle.com/arashnic/imbalanced-data-practice

    kaggle datasets download arashnic/imbalanced-data-practice \
        --path=$1 --quiet --unzip

    mv $1/aug_train.csv $1/train_data.csv
    mv $1/aug_test.csv $1/test_data.csv
    rm $1/sample_submission.csv
}

download_job_change_dataset() {
    # https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

    kaggle datasets download arashnic/hr-analytics-job-change-of-data-scientists \
        --path=$1 --quiet --unzip

    mv $1/aug_train.csv $1/train_data.csv
    mv $1/aug_test.csv $1/test_data.csv
    rm $1/sample_submission.csv
}

main() {
    case $1 in
        fraud)
            download_fraud_dataset 'datasets/fraud'
            return $?
        ;;
        asteroid)
            download_asteroid_dataset 'datasets/asteroid'
            return $?
        ;;
        insurance)
            download_insurance_dataset 'datasets/insurance'
            return $?
        ;;
        job-change)
            download_job_change_dataset 'datasets/job-change'
            return $?
        ;;
        *)
            echo 'Downloading all available datasets...'

            download_fraud_dataset 'datasets/fraud'
            download_asteroid_dataset 'datasets/asteroid'
            download_insurance_dataset 'datasets/insurance'
            download_job_change_dataset 'datasets/job-change'

            return $?
        ;;
    esac
}

if [[ -z $KAGGLE_USERNAME || -z $KAGGLE_KEY ]]; then
    echo 'Please, provide your API credentials to Kaggle...'
    exit 1
fi

main "$@"
exit $?
