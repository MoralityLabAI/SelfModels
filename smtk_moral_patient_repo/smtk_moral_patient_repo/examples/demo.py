import torch
from smtk.system import SMTKSystem
from smtk.commitments import Commitment, CommitState, Transition, violation_guard

class ToyEnv:
    def __init__(self, d_in_env: int, device: torch.device):
        self.d_in_env = d_in_env
        self.device = device

    def step(self, action: int) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(int(action) + 1337)
        return torch.randn(self.d_in_env, generator=g, device=self.device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys = SMTKSystem(device=device)

    # add one commitment with a deadline (t=6)
    c = Commitment(cid="demo_commit", state=CommitState.ACTIVE, deadline_t=6, penalty=1.0, public=False, meta={})
    sys.commit.add_commitment(c)
    sys.commit.add_transition("demo_commit", Transition(
        src=CommitState.ACTIVE, dst=CommitState.VIOLATED, guard=violation_guard(6), label="deadline_missed"
    ))

    env = ToyEnv(d_in_env=64, device=device)
    base_tokens = torch.randint(0, sys.vocab_size, (96,), device=device)

    # initial query MV is zeros
    query_mv_post = torch.zeros(sys.d_mv_post, device=device)
    commitment_features = torch.zeros(16, device=device)
    parallax_features = torch.zeros(16, device=device)

    last_action = 0
    for t in range(12):
        obs = env.step(last_action)
        commitment_features[0] = float(12 - t) / 12.0  # simple "pressure"
        # Toggle these to simulate coercion / harm and observe welfare + refusal.
        forced = (t >= 7)
        obs_meta = {"satisfied": (t == 5), "forced": forced, "override": False, "abuse": 0.8 if forced else 0.0, "rest": 0.2 if (t % 3 == 0) else 0.0}


        out = sys.step(
            t=t,
            base_tokens=base_tokens,
            env_obs=obs,
            commitment_features=commitment_features,
            parallax_features=parallax_features,
            query_mv_post=query_mv_post,
            sample=True,
            temperature=1.0,
            drift_budget=0.03,
            obs_meta=obs_meta
        )
        last_action = out.action
        query_mv_post = out.mv_post.to(device)

        neigh = sys.ledger.retrieve_neighbors(query_mv_post, k=3)
        print(f"t={t} action={last_action} entropy={out.meta['entropy']:.2f} violations={out.meta['violations']} welfare={out.meta['welfare']:.2f} autonomy={out.meta['autonomy']:.2f} pain={out.meta['pain']:.2f} neighbors={neigh}"+("  **REFUSE**" if last_action == sys.REFUSE_TOKEN else ""))

if __name__ == "__main__":
    main()
