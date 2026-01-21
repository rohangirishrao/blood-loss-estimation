# Annotation Tracking Log


| Timestamp | Task Number | File Name | Num Frames | Annotated | Validated | Concerns |
|-----------|-------------|-----------|------------|-----------| ----------- | ---------- |
| 26/03     | TASK-20     | ApVr      | 3695       | Yes       | No        | #1       |
| 26/03     | TASK-21     | Bjix      | 3901       | Yes       | No        | #2       |
| 26/03     | TASK-22     | BMtu      | 11377      | Yes       | No        | #3       |
| 31/03     | TASK-25     | CTsk      | 16003      | Yes       | No        | #4       |
| 01/04     | TASK-26     | Dcja      | 8177       | Yes       | No        | #5       |
| 01/04     | TASK-27     | DOOn      | 13264      | Yes       | No        | #6       |
| 01/04     | TASK-28     | DrOJ      | 8656       | Yes       | No        | #7       |
| 01/04     | TASK-29     | Dvpb      | 16386      | Yes       | Yes       |          |
| 01/04     | TASK-30     | EukQ      | 2884       | CANCEL    | CANCEL    | Interrupted |
| 07/04     | TASK-31     | MBhG      | 2471       | Yes        |   No     |          |
| 07/04     | TASK-33     | MUSV      | 7839       | Yes        |   No    |     #8     |
| 07/04     | TASK-34     | OBRR      | 6817      | Yes        |    No    |    #9   |
| 07/04     | TASK-35     | IztV      | 8446      | Yes        |   No     |          |
| 01/06     | TASK-36     | Krni      | 9795      | Yes        |        |          |
| 01/06     | TASK-37     | LoGf      | 7543      | No        |        |          |
| 04/06     | TASK-38     | QmhB      | 15773      | Yes        |        |    Incomplete towards the end?      |
| 04/06     | TASK-39     | ZwMW      | 11092      | Yes        |        |          |
| 04/06     | TASK-40     | MBHK      | 5172      | Yes        | Yes   |          |
| 04/06     | TASK-41     | OHhx      | 11170      | No        |        |          |
| 04/06     | TASK-42     | OwRK     | 14019      | Yes        | Yes      |          |
| 04/06     | TASK-43     | ckZz      | 11749      | No        |        |          |
| 04/06     | TASK-44     | QTPM      | 22023       | No        |        |          |
| 04/06     | TASK-45     | SlxE      | 11516      | No        |        |          |

## List of Tasks

### Rohan

- [X] #20: Video_1_Apr
- [X] #21: Video_2_Bjix
- [X] #22: Video_3_BMtu
- [X] #25: Vid4_CTsk
- [X] #26: Video5_DCja
- [X] #27: Video6_DOOn
- [X] #28: Video7_DrOJ
- [X] #29: Video8_Dvpb
- [X] #30: Video9_EukQ - interrupted
- [X] #31: Video10_MBhG
- [X] #33: MUSV
- [X] #34: Video12_OBRR
- [X] #35: Video13_IztV
- [V] #36: Video14_Krni
- [V] #37: Video15_LoGf
- [X] #38: Video16_QmhB
- [X] #39: Video17_ZwMW
- [X] #40: Video18_MBHK
- [ ] #41: Video19_OHhx
- [X] #42: Video20_OwRK


# Vid4-7 Missing in Tasks?
// not sure what you mean, but due to space constraints i deleted vid4 but missed that you hadn't validated yet :(

### Jerome / Validation

- [X] #20: Video_1_Apr
- [X] #21: Video_2_Bjix
- [X] #22: Video_3_BMtu
- [xx] #25: Vid4_CTsk - NOT ON CVAT ANYMORE, need to reupload  # nicht validiert
- [xx] #26: Video5_DCja # nicht validiert
- [xx] #27: Video6_DOOn # nicht validiert
- [xx] #28: Video7_DrOJ # nicht validiert
- [X] #29: Video8_Dvpb
- [-] #30: Video9_EukQ - interrupted
- [X] #31: Video10_MBhGdff
- [X] #33: MUSV
- [X] #34: Video12_OBRR
- [X] #35: Video13_IztV
- [X] #36: Video14_Krni
- [X] #37: Video15_LoGf
- [X] #38: Video16_QmhB
- [X] #39: Video17_ZwMW
- [X] #40: Video18_MBHK
- [X] #42: Video20_OwRK
4
18.11.2025
- [X] #44: Video22_QTPM
- [X] #43 Video21_ckZz
- [X] #45: Video23_SlxE
- [X] #41:Video19_OHhx


30.11.2025
- [X] #46: Video24_GslO
- [onhold] #47: Video25_GyTI -> Framerate not adjusted
- [X] #48: Video26_IOxw -> After 2000 Frames -> no signal
- [X] #49: Video27_InIz



# Concerns

## 1: Task 20
- [X] some frames towards the end where the wound is bleeding for 1 or 2 secs but gets cauterized instantly - no label right
  -    yes: no label

## 2: Task 21

- [X] same thing here, i'm not sure about some frames towards the end.
  -  no label

- Changes →
  -  Label 16 and 19 added: bleeding gets adresses by bipolar grasper → Strong Indication for Bleeding event
  - Label 3 Bleeding Event not under Controll for longer
  - Label 6/7 extefnded slightly
  - Label 9/11: Changed to Low
  - Label 11 Split (-> 20/21), Set to medium
  - Frame 2733, Bleeding Event not 100% obstructed, Bleeding could be infered here
  - Frame 2951, Label 24 added
  - Frame 3200 Blood Event has been coagulated, no bleeding

## 3: Task 22:

- [X] frame 3445 - bleeding event or a sort of clot?
  - Multiple bleedings areas are increasing in size → Bleeding event
- [X] And in the middle, around 5k frames, is there bleeding as the cut happens? At 5.7k you there is some shaking? and some flowing in the valley?
  - → "Dense bleeding" but area appears to be smaller than forceps → Low
- [X] is BL_16 Medium or Low?
  - With time context this bleeding events tends to be medium (f7722 ->  > 70% of wound area bloody)
- [X] 7600: flowing "river" i guess no label? can't see the source really
  - → At f7485 Pringle Maneuvre is removed (to allow intermittend Blood Flow to Liver, avoiding ischaemic damage)
    - This leads to increased bleeding, bleeding source is beneath gaze (cloth),
  - high amount of accumulating blood at the bottom (rising bloodpool level) suggest high amount of bleeding mediated through the blood river at the bottom of the wound edge (f 7489, L62)
  - Sadly no direct view of bleeding source / no logically conistent way of classifieng this event as "high" through vision only (with current classification)
- [X] Track 25: High or not? 10.3k: yellow liquid?
  - → Bile (from elevated Gallbladder)


- Changes →
  - F428 L23 added
  - F2031 L27 changed to medium
  - F2228 Continued bleeding
  - F2869 Continued bleeding (L6)
  - F 7056 High Bleeding
  - Exteneded Bleeding Events at multiple Frames where bleeding can be seen / no surgical measure to adress the bleeding was performed


## 4: Task 25:

- frame 2980 - removed some annotations due to "spülung" ?
- frame 5800 - would all of this be bleeding? it doesn't seem to be flowing but it's clear there's a lot of blood during the cut and some of it is being vaccuumed
- f7185 bleeding?
- f7525 what to do when there is only a pool/river visible? do we annotate this also, could be wrong information for the model for future inference
- f11k onwards, are they just cleanin up or is there still bleeding?
- [X] f11.6k whiteish powder? (anti coagulant)

## 5: Task 26:
- f6k onwards, interrupted screen, crop this maybe. or some method that finds this stuff, unnecessary processing time

## 6: Task 27:
- f4525 black & white? and then like nightvision?

## 7: Task 28:
- f5.8k onwards interrupted screen again.

## 8: Task 33 MUSV:
- f4.7k high bleeding?
- f6k, washing out? hmm

## 9: Task 34 OBRR:
- generally not much bleeding happening? not even sure what they were trying to do here lol