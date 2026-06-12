# OBJ Head Upside Down — Analysis and Fix

## Root Cause

The OBJ model has the face at high Y values and the top of the head at low Y values — this means the model's native coordinate system in the `.obj` file already has the head upside down relative to screen space. When the projection math computes the screen Y coordinate, it keeps this inversion.

Currently the `project` function in `WireframeModel` returns:
```typescript
y: pitchY * 110 / depth + bob,
```

The `pitchY` value preserves the model's inverted Y. For example:
- Face vertex (Y=3.27 in OBJ) → screen Y ≈ 113 (bottom of canvas)
- Top-of-head vertex (Y=1.52 in OBJ) → screen Y ≈ 45 (top of canvas)

This puts the face at the bottom and top-of-head at the top — the head is upside down.

## How to Fix It

**One line change** in `src/App.tsx` inside the `project` function of `WireframeModel`:

Current (line ~220):
```typescript
y: pitchY * 110 / depth + bob,
```

Change to:
```typescript
y: -(pitchY * 110 / depth) + bob,
```

The **minus sign** flips the Y coordinate, inverting the head vertically. The face (high model Y) will go to the top of the canvas, and the top of the head (low model Y) will go to the bottom — making it right-side up.

## Why This Works

The canvas coordinate system has Y going **down** (0 at top, increasing downward). The OBJ model's "top of head" is at low Y values and "face/chin" is at high Y values. Negating the projected Y reverses this, putting the model's high values (face) at the top of the screen (low canvas Y) and low values (top of head) at the bottom (high canvas Y). Combined with the 180° yaw rotation (which faces the head toward you), this gives you a head that looks at the user and bobs right-side up.