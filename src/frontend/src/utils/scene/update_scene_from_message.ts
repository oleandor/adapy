import {Message} from "../../flatbuffers/wsock/message";
import {useModelStore} from "../../state/modelStore";

export const update_scene_from_message = (message: Message) => {
    console.log('Received scene update message from server');
    let scene = message.scene();
    if (!scene) {
        console.error("No scene object found in the message");
        return;
    }
    let fileObject = scene.currentFile();
    if (!fileObject) {
        console.error("No file object found in the message");
        return;
    }
    // Get the filedata array (this is typically a Uint8Array)
    let data = fileObject.filedataArray();

    if (!data) {
        console.error("No filedata found in the file object");
        return;
    }

    const blob = new Blob([data], {type: 'model/gltf-binary'});
    const url = URL.createObjectURL(blob);

    useModelStore.getState().setModelUrl(url, null, ""); // Set the URL for the model
}