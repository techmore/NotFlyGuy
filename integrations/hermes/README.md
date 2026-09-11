# Hermes / Signal connection

The native Hermes plugin registers `/fly status` and `/fly <goal>`. Hermes
uses its existing Signal account, chat authorization, and reply delivery.
The plugin calls the habitat at http://127.0.0.1:8765; no Signal credentials
or group identifiers belong in this repository.

Install by linking `notflyguy/` into the active Hermes profile's `plugins/`
directory, running `hermes --profile PROFILE plugins enable notflyguy`, and
restarting its gateway. Built-in tool override permission is not required.

Local setup: enabled in the default and `not-rocky` profiles. The existing
gateway routes Signal through `not-rocky`, whose Signal adapter was enabled.

Verification: the plugin passed Hermes Plugin Doctor; the running gateway
reported Signal connected. A direct plugin call retrieved a current FlyBody
image and timestep. Actual incoming Signal command and returned reply still
require an end-to-end check with `/fly status` in the existing chat.

Commands `/fly pause`, `/fly resume`, and `/fly reset` queue operations on the
MuJoCo main thread. These were verified against the live simulator. The
profile's real Hermes command registry also resolved `/fly status` and returned
a live timestep and image status. This does not replace Signal delivery testing.

The profile now uses its existing local MLX model on port 8083 because the
previous Nous provider lacked authentication. A local completion was verified.
That MLX process and the habitat currently run in terminal sessions; automatic
startup/recovery has not been established for them.

The localhost viewer URL works on the Mac running the habitat. Remote browser
access is not configured. Goals are recorded, but do not yet produce trained
motor behavior. Language responses come from Hermes or this command handler;
there is no connectome language model.
