# Crash recovery and module rejoin

This is the target protocol contract for new communication interfaces and the
sensor migration in [DV-SENSORS.md](../DV-SENSORS.md). It does not claim that
current modules implement automatic recovery. Existing transport is described
in [modcom.md](modcom.md); clock behavior is described in [clock.md](clock.md).

Protocols must permit a module to restart, rediscover peers, reconcile current
state, and rejoin an ongoing mission. Automatic mission recovery, persistent
checkpoints, and module-specific recovery policy may be implemented later.
A module may initially reconnect and report that intervention is required.
Reconnection alone must neither cancel a mission nor authorize motion.

## Identities and continuity

| Identity | Meaning and lifetime |
|---|---|
| `vehicle_id` | Stable identity of the simulated or physical vehicle; survives bridge restarts. An instance discovery address alone is not proof of vehicle identity. |
| `provider_session_id` | Fresh UUID for each provider process lifetime. Identifies the publisher, not a new vehicle or mission. |
| `module_session_id` | Fresh UUID for each consumer/controller process lifetime; distinguishes a restarted module from its former instance. |
| Sensor `generation` | Configuration/transport revision within a provider session; changes after successful profile Apply. |
| `clock_domain_id`, `clock_epoch` | Identity of the authoritative time source and its continuity epoch. Epoch changes when continuity cannot be established. |
| Sensor `reset_epoch` | Invalidates sensor history after a reset, without implicitly resetting mission state. |
| `mission_id` / existing `run_id` | Explicit execution identity independent of process sessions, sensor generations, and clock epochs. |

A physical-drone bridge must verify which vehicle it reconnected to before
advertising readiness. An unexpected vehicle identity requires explicit
reconciliation, not automatic reuse of the previous mission or control lease.
Only one active provider may own an instance. Restart cleanup must not replace
a live provider; new provider implementations must define exclusive ownership
and how stale ownership is detected before claiming readiness.

A provider restart need not restart vehicle time. A bridge may preserve its
clock domain/epoch only when it can establish continuity from the authoritative
source or durable state. Otherwise it publishes a new epoch. A DSIM process
restart starts a new clock epoch; a drone reset preserves simulated time.
Consumers must not compute intervals across clock epochs. Sample timestamps
identify their clock epoch, and discovery identifies the clock domain.

## Recoverable discovery and samples

The discovery address is stable, while data-area names include provider session
and generation. Create and initialize new areas before committing a complete
manifest. Publish the committed manifest atomically and define a coherent read
operation; separate reads of individual keys are not an atomic snapshot.

A consumer must retry opening the stable discovery name after heartbeat expiry,
read failure, or explicit provider replacement. An old shared-memory mapping
can remain readable after unlink, so readable data and unchanged generation
are not evidence of a live publisher. Discovery retries use bounded backoff
and wall time; all data cadence continues to use the declared data clock.

After discovering a changed session or generation, a consumer:

1. Validates vehicle identity, schemas, capabilities, and committed manifest.
2. Closes old data handles and clears transport/association caches.
3. Opens advertised areas, retrying discovery if another rollover races opening.
4. Obtains current state and waits for the required readiness level.
5. Reconciles its execution state before attempting to resume work.

Sensor records are identified by session, generation, sensor ID, and sequence.
Camera metadata additionally matches the exact committed video sequence.
Sequence reuse across sessions is safe because sessions are distinct. Within a
session/generation, drone reset preserves transport sequences and capture IDs
and increments `reset_epoch`. Consumers invalidate sensor history on that
change; mission policy is separate. Rings provide bounded recent history, not
replay of all measurements missed during downtime. Consumers tolerate gaps and
explicitly report missing inputs or the need to rebuild estimation state.

A consumer restart follows the same discovery process without requiring a
provider restart or special acknowledgment. Its new module session advertises
fresh subscriptions and health. Old module-session health expires independently.

## Readiness and current-state reconciliation

Expose separate facts for process liveness, vehicle connection, state
synchronization, and readiness for control. An alive bridge may still be
reconnecting. A ready sensor publisher does not imply that control is ready.
Report recovery progress and an explicit reason when continuation is blocked.

Recovery must not depend on receiving past events. Publish retained snapshots
or queryable state for the vehicle identity and pose, armed/mode state,
capabilities, control ownership, active mission/run identity, and execution
progress where supported. Identify the session and revision associated with a
snapshot; mark unavailable or unknown information explicitly. Events notify
changes; snapshots allow a late joiner to establish the present state.

A future recovering module compares this state with its persisted or otherwise
recoverable mission state. It may continue, rebuild local estimation, request
intervention, or decline resumption. Sensor/profile changes can invalidate an
algorithm's assumptions without terminating the mission for all other modules.
The protocol supports this decision; it does not choose one universal policy.

## Commands, leases, and unknown outcomes

Sensor samples may be dropped. Commands require explicit retry semantics.
New control contracts must carry a request ID, sender module session, target
provider session, applicable vehicle/mission identity, and an expiration with
a declared clock basis. Leased commands carry an ownership token scoped to the
current provider session. Reject expired commands, old-session commands, and
stale ownership tokens. A restarted controller must reconcile ownership and
obtain a valid lease before issuing leased commands.

Define command outcomes including rejected, accepted/in progress, completed,
failed, and unknown. A receipt acknowledgment is not evidence of completion.
Provide correlated results and current-state queries through pymembus. Specify
a bounded deduplication/result-retention window and its behavior after expiry.
A retry with the same request ID within that window must not execute twice.
Do not claim exactly-once execution across crashes without durable evidence.

If a bridge crashes after forwarding an action but before recording its result,
the outcome is unknown. On rejoin, query actual vehicle state and reconcile
before retrying. Do not replay stale motion setpoints or blindly repeat an
uncertain takeoff, land, or mission-start action. A future bridge may recover
an outcome from vehicle-side command/mission identity or a durable journal;
otherwise it reports uncertainty. Existing explicitly lease-free emergency
commands keep their documented policy; they are not a general replay exception.

Lease loss and disconnection invoke the vehicle/module's defined failsafe
behavior. Reconnecting does not automatically restore old authority. The
physical vehicle may have continued its onboard mission while the bridge was
absent, so bridge-local state must not be treated as authoritative flight state.

## Required now and deferred implementation

During the sensor contract phase, freeze restart-safe identities, atomic
rediscovery, stale-handle detection, clock/reset epochs, and reconnectable
subscription semantics. When control protocols are revised, preserve the
command and reconciliation requirements above. Document unsupported recovery
capabilities explicitly rather than implying automatic continuation.

Future module work may add persisted mission checkpoints, durable command
journals, automatic estimator rebuilding, and automatic mission resumption.
No DFGB changes are required by this document or the sensor milestone.

Protocol tests should cover producer restart while consumers stay alive,
consumer restart while the provider stays alive, stale readable mappings,
rollover during opening, missing recent samples, and clock discontinuities.
Control recovery tests must cover stale leases, expired/repeated commands, and
a crash between vehicle execution and acknowledgment. Verify that rejoining
does not itself cancel a mission, replay motion, or claim recovered authority.
