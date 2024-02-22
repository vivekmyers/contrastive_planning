SHELL = /bin/bash

RETRY = 5
RETRY_CMD = $(shell test $(RETRY) -gt 0 \
	&& echo '|| { echo "$$(tput setaf 1)Retrying $@ ($(shell expr $(RETRY) - 1) retries left)...$$(tput sgr0)" && exec $(MAKE) $@ RETRY=$(shell expr $(RETRY) - 1) ; }' \
	|| echo '|| echo "$$(tput setaf 1)Giving up on $@...$$(tput sgr0)"' \
)
PREPARE = rm -f $(CONDA_PREFIX)/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock.lock

NSEED = 100
SEEDS = $(shell seq $(NSEED))
TRAIN_DATA = contrastive.pkl vip.pkl pca.pkl none.pkl train_dataset.pkl val_dataset.pkl
MODELS = $(foreach data,$(TRAIN_DATA),$(foreach seed,$(SEEDS),checkpoints/%/$(seed)/$(data)))
SAMPLES = $(shell seq 10)

BARWPT = 1
LWPT = 5
RWYPT = 20
BARPLOT = --barplot --n_wypt $(BARWPT)
LINEPLOT = --waypoint_mse --n_wypt $(LWPT)
SCOREPLOT = --rollout
MSE = --mse --trials 200

ifdef DEBUG
RETRY = 2
NSEED = 10
XTRAINARGS = --train_steps 100 --num_shuffle 1
XEVALARGS = --trials 2 --jobs 5
SAMPLES = $(shell seq 2)
endif

all: maze2d-large-v1 door-human-v0 ;

maze2d-large-v1: figures/maze2d-large-v1/rollout_scores.pdf ;

door-human-v0: figures/door-human-v0/waypoint_mse.pdf figures/door-human-v0/plan_mse.pdf figures/door-human-v0/samples ;

figures/%/samples: $(addprefix figures/%/samples/plan_embedding,$(addsuffix .pdf,$(SAMPLES))) | figures/%/ ;

define sample_rule
figures/%/samples/plan_embedding$(1).pdf: figures/%/ $(MODELS)
	CUDA_VISIBLE_DEVICES= python run_plot.py --env $$* --seed $(1) --plot_plan $$(RETRY_CMD)
endef

$(foreach sample,$(SAMPLES),$(eval $(call sample_rule,$(sample))))

figures/%/rollout_scores.pdf: results/%/rollout_scores.pkl | figures/%/ 
	CUDA_VISIBLE_DEVICES= python run_plot.py --env $* $(SCOREPLOT) $(RETRY_CMD)

figures/%/waypoint_mse.pdf: results/%/plan_mse_$(LWPT).pkl | figures/%/ 
	CUDA_VISIBLE_DEVICES= python run_plot.py --env $* $(LINEPLOT) $(RETRY_CMD)

figures/%/plan_mse.pdf: results/%/plan_mse_$(BARWPT).pkl | figures/%/ 
	CUDA_VISIBLE_DEVICES= python run_plot.py --env $* $(BARPLOT) $(RETRY_CMD)

figures/%/:
	mkdir -p $@

results/%/rollout_scores.pkl: $(MODELS) | results/%/
	$(PREPARE)
	CUDA_VISIBLE_DEVICES= python run_eval.py --env $* --rollout --n_wypt $(RWYPT) $(XEVALARGS) $(RETRY_CMD)

results/%/plan_mse_$(LWPT).pkl: $(MODELS) | results/%/
	$(PREPARE)
	CUDA_VISIBLE_DEVICES= python run_eval.py --env $* $(MSE) --n_wypt $(LWPT) $(XEVALARGS) $(RETRY_CMD)

results/%/plan_mse_$(BARWPT).pkl: $(MODELS) | results/%/
	$(PREPARE)
	CUDA_VISIBLE_DEVICES= python run_eval.py --env $* $(MSE) --n_wypt $(BARWPT) $(XEVALARGS) $(RETRY_CMD)

results/%/:
	mkdir -p $@

define train_rule
$(addprefix checkpoints/%/$(1)/,$(TRAIN_DATA)): checkpoints/%/$(1) ;
checkpoints/%/$(1): | checkpoints/%/
	$(PREPARE)
	CUDA_VISIBLE_DEVICES=$(shell nvidia-smi --query-gpu=index --format=csv,noheader | shuf | awk NR==1) \
	python run_train.py --env $$* --seed $(1) $(XTRAINARGS) $$(RETRY_CMD)
endef

checkpoints/%/:
	mkdir -p $@

$(foreach seed,$(SEEDS),$(eval $(call train_rule,$(seed))))

clean:
	rm -rf figures results checkpoints
	$(PREPARE)

.SECONDARY:

