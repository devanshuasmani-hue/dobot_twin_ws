# Phase 3 - physical Dobot synchronisation

## What you get
* **Physical twin** (orange label, `/real/...`): shows the robot's *actual* joint angles in real time (25 Hz).
  Never clamped, never smoothed. When data stops, the label says `NO DATA` (the model freezes - it never pretends).
* **Simulated twin** (blue label, `/sim/...`): unchanged. Used to *preview* a move before it goes to the hardware.
* The robot can also be commanded from the twin, behind several safety gates.

## First-time setup (once)
```bash
sudo apt install python3-serial python3-tk
sudo usermod -aG dialout $USER        # then LOG OUT and back in (needed to open /dev/ttyUSB0)
```
Power the Dobot, connect USB. Check the port: `ls /dev/ttyUSB* /dev/ttyACM*`.

## Step 1 - read-only connection test (cannot move anything)
```bash
dobot probe                     # prints live x,y,z,r and J1..J4; move the arm by hand and watch the numbers
```
If it prints `ERROR: no answer`, tell me the output of `ls /dev/ttyUSB* /dev/ttyACM*` and `dmesg | tail`.

## Step 2 - mirror the real robot (still read-only)
```bash
dobot full                      # or: dobot full port:=/dev/ttyUSB0
```
Drag the real arm by hand (hold the unlock button): the orange twin must follow. Status shows `UNCALIBRATED (read-only)`.

## Step 3 - hands-only calibration (still no motion commands)
```bash
dobot calibrate
```
Move the arm by hand to 6 different poses (vary the rear arm J2 and forearm J3 a lot), pressing Enter at each.
It determines whether your robot's J3 is relative or absolute, and the offset of the robot's X/Y/Z reading,
and saves `~/.dobot_twin/calibration.yaml`. It refuses to save if the fit is poor or ambiguous.
Restart `dobot full` afterwards.

## Step 4 - commanding the robot (only after steps 1-3 look right)
In the **PHYSICAL ROBOT** window: type a target -> *Preview in SIM* -> look at the blue twin ->
tick **ARM** -> *EXECUTE on REAL* -> confirm. Speed defaults to 20 % (hard cap 50 %).
Rejected commands show the reason. **STOP** always works; ARM auto-expires after 60 s (and stops the robot).
Keep a hand on the robot's physical stop/unlock button and the workspace clear.

## Safety gates (all must hold for any real motion)
calibrated - armed (< 60 s old) - fresh pose (< 0.5 s) - no robot alarm - goal inside reach, joint limits and
floor + 10 mm margin, including the intermediate path. Losing the pose stream while armed stops + disarms.
Connecting and polling send **read requests only** (verified by an automated test).

## Not verified without hardware (please report anything odd)
* The protocol command IDs / payload layouts follow the published Dobot protocol; they have been tested only against
  the built-in simulated robot. Reads (pose/alarms) are tried first for exactly this reason.
* Gripper: the Dobot gripper is pneumatic (open/close only). If open/close are swapped, set `gripper_inverted: true`
  in `config/driver.yaml`. The twin shows the *reported* state when available.
* Dobot's own coordinate origin differs from the twin's - handled by calibration.
