
#TODO Video10_MBhG überprüfen, scheint unvollständig

## Current Approach

### Bleeding Event Characterization

Bleeding events refere to suspected sources of *active* bleeding + associated attributes like flowing "blood streets"

#### Label: Characterizes Flow estimation

- Low:
	- Every bleeding event where one can see underlyieng tissue
	- Everything not medium or high
- Medium:
	- Area bigger than Forceps in which bleeding covers > 70% of visible wound area
	- Persistent Bleeding over atleast 3s
- High:
	- Any Area with: Visible Flow / Currents / Splashes
	- Persistent Bleeding over atleast 3s

#### DOs / DONTs
- Avoid Segmenting Bloodpools / focus on wound area

- Only sitch label when change is persistent over ~10s, or bleeding event changes to high flow
	- Avoid frequent switching between labels

- If bleeding event is obstruced for more than 5s change mask

- Shapes: rectangle / ellipse

- Generally avoid annotating events which are not adressed surgically at all.



## Additional/Future Approaches

### Bleeding Events

#### Blood Streets
- Flowing
- Stagnant
#### Pooling
- Raising
- Stagnant
#### Dilution Level
- No Dilution
- Mild Dilution
- Strong Dilution

### Phase Annotation



### Field of View Size
