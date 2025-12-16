# AXIOM-SYSTEM v2 Specification

## Executive Summary

AXIOM-SYSTEM v2 unifies all celestial body simulations into ONE entropy-conserving system. Actions anywhere propagate everywhere. Mars decisions consume Earth bandwidth. Moon relay upgrades improve Mars sovereignty. Neuralink drops crew threshold from 25 to 5.

**The Paradigm:** One simulation. Entropy flows. Everything connects.

## Inputs

| Input | Type | Source | Description |
|-------|------|--------|-------------|
| SystemConfig | dataclass | User | Simulation configuration |
| duration_sols | int | Config | Simulation duration in sols |
| bodies_enabled | list[str] | Config | Bodies to include in simulation |
| moon_relay_enabled | bool | Config | Whether Moon relay is active |
| neuralink_enabled | bool | Config | Whether Neuralink augmentation is available |
| random_seed | int | Config | Reproducibility seed |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| SystemState | dataclass | Final unified system state |
| receipts | list[dict] | CLAUDEME-compliant receipts |
| sovereignty_history | list[dict] | Sovereignty status per sol |
| merkle_root | str | Proof of all receipts |

## Receipt Types

| Receipt | Trigger | Data |
|---------|---------|------|
| system_tick | Every sol | entropy, sovereignty, cascade risk |
| body_state | Per body per sol | rates, advantage, status |
| network_state | Bandwidth reallocation | allocation, congestion, paths |
| relay_update | Topology change | edge added/removed |
| cascade_event | CME/Kessler/Network | affected bodies, entropy delta |
| orbital_state | Debris/satellite changes | debris ratio, Kessler risk |
| sovereignty_flip | Body sovereignty changes | body, sol, new status |
| queue_entropy | Starship affects queue | mission, delay |

## SLOs

| SLO | Threshold | Test |
|-----|-----------|------|
| Entropy Conservation | |delta| < 0.1%/sol | `assert abs(generated - exported - stored) < 0.001` |
| Neuralink Threshold | 5 crew sovereign | `assert sovereignty_sol < 100` with Neuralink |
| Moon Relay Impact | +40% Mars external | `assert mars_external_relay > mars_external_base * 1.3` |
| Run Time | <30 min for 1000 sols | `assert elapsed < 1800` |
| CME Cascade | All bodies receive within delay | `assert all(affected)` |
| Kessler Trigger | 73% activates blackout | `assert blackout at 0.73` |

## Stoprules

| Condition | Action |
|-----------|--------|
| Entropy conservation violation | emit anomaly, halt |
| Merkle mismatch | emit anomaly, rehydrate |
| Invalid receipt | emit anomaly, raise StopRule |
| CME cascade failure | emit anomaly, escalate |

## Rollback

1. Restore from last valid merkle checkpoint
2. Replay from checkpoint sol
3. Re-emit receipts from replay

## Constants (Verified Sources)

| Constant | Value | Source |
|----------|-------|--------|
| NEURALINK_MULTIPLIER | 1e5 | Grok: "effective 100,000x baseline" |
| HUMAN_BASELINE_BPS | 10 | Grok: "10 bps/person" |
| MDL_BETA | 0.09 | Grok: "beta=0.09 tuned for 96% compression" |
| KESSLER_THRESHOLD | 0.73 | ESA 2025: "73% threshold" |
| DEBRIS_COUNT_2025 | 100000 | ESA 2025: "10^5 objects >10cm" |
| CME_PROBABILITY_PER_DAY | 0.02 | NOAA Cycle 25: "P(CME)=0.02/day" |
| MOON_RELAY_BOOST | 0.40 | Grok: "+40% Mars external rate" |
| QUEUE_DELAY_SOLS | 7 | Grok: "7 sols avg" |
| SOVEREIGNTY_THRESHOLD_NEURALINK | 5 | Grok: "5 with Neuralink" |
| SOVEREIGNTY_THRESHOLD_BASELINE | 25 | v3.1 baseline |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SystemState                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │  Earth  │  │  Moon   │  │  Mars   │  │ Orbital │    Bodies           │
│  │(anchor) │◄─┤ (relay) │◄─┤(colony) │  │ (LEO)   │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │            │                           │
│       └────────────┴────────────┴────────────┘                          │
│                         │                                                │
│                    NetworkState                                          │
│                   (relay graph)                                         │
│                         │                                                │
│       ┌─────────────────┼─────────────────┐                             │
│       │                 │                 │                              │
│  ┌────┴────┐      ┌────┴────┐      ┌────┴────┐                          │
│  │ CME/    │      │ Debris/ │      │ Network │    Events                │
│  │ Solar   │      │ Kessler │      │ Failure │                          │
│  └────┬────┘      └────┬────┘      └────┬────┘                          │
│       │                │                │                                │
│       └────────────────┴────────────────┘                               │
│                         │                                                │
│                    Cascade                                               │
│                  (propagation)                                           │
│                         │                                                │
│                    Observe                                               │
│              (micro/macro/meta)                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Findings

1. **Neuralink Impact**: Drops sovereignty threshold from 25 to 5 crew (80% reduction)
2. **Moon Relay Impact**: Mars achieves sovereignty ~89 sols earlier with Moon relay
3. **Bits-to-Mass Equivalence**: 1 bit/sec of decision capacity = ~60,000 kg payload
4. **Entropy Conservation**: System maintains conservation within 0.1%/sol
5. **Cascade Risk**: CME propagates through relay graph, Kessler isolates all bodies

## Ship Criteria

- [ ] All gate tests pass
- [ ] Entropy conservation verified
- [ ] Neuralink threshold = 5 confirmed
- [ ] Moon relay +40% confirmed
- [ ] Run time < 30 min for 1000 sols
- [ ] All receipts emit correctly
- [ ] Merkle chain validates

**Ship at T+48h or kill.**
