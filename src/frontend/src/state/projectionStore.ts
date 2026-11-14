// The 3D view's projection: perspective (the default) or orthographic. One setting for the
// page, remembered in this browser through the store's `persist` middleware. The three.js
// viewport follows it (utils/scene/orthographicProjection.ts). The three.js-free viewport says
// it has no orthographic view yet (`ViewportApi.unavailable("orthographic")`), and the command
// is offered disabled there. No three.js here, so that chunk may import it.

import {create} from "zustand";
import {persist} from "zustand/middleware";

/** The localStorage entry the setting is kept under. */
export const PROJECTION_STORAGE_KEY = "ada:camera-projection";

interface ProjectionState {
    isOrthographic: boolean;
    setOrthographic(on: boolean): void;
    toggleOrthographic(): void;
}

export const useProjectionStore = create<ProjectionState>()(
    persist(
        (set, get) => ({
            isOrthographic: false,
            setOrthographic: (on) => set({isOrthographic: on}),
            toggleOrthographic: () => set({isOrthographic: !get().isOrthographic}),
        }),
        {
            name: PROJECTION_STORAGE_KEY,
            version: 1,
            partialize: (s) => ({isOrthographic: s.isOrthographic}),
        },
    ),
);

/**
 * Call `apply` with the projection now, and again each time it changes; returns the
 * unsubscribe. How a viewport keeps its camera in step with the setting.
 */
export function followProjection(apply: (isOrthographic: boolean) => void): () => void {
    apply(useProjectionStore.getState().isOrthographic);
    return useProjectionStore.subscribe((now, before) => {
        if (now.isOrthographic !== before.isOrthographic) apply(now.isOrthographic);
    });
}
