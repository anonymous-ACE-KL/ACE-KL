for idx in $(seq 0 1 1);
do
	python explain_user_ae_netflow.py 1 netflow_train.pkl netflow_test.pkl $idx
done
