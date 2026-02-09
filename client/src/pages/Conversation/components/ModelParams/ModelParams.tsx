import { FC, RefObject } from "react";
import { useModelParams } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";

type ModelParamsProps = {
  isConnected: boolean;
  isImageMode: boolean;
  modal?: RefObject<HTMLDialogElement>,
} & ReturnType<typeof useModelParams>;
export const ModelParams: FC<ModelParamsProps> = ({
  textTemperature,
  textTopk,
  audioTemperature,
  audioTopk,
  padMult,
  repetitionPenalty,
  repetitionPenaltyContext,
  imageResolution,
  voicePrompt,
  textPrompt,
  setTextTemperature,
  setTextTopk,
  setAudioTemperature,
  setAudioTopk,
  setPadMult,
  setRepetitionPenalty,
  setRepetitionPenaltyContext,
  setImageResolution,
  setVoicePrompt,
  setTextPrompt,
  resetParams,
  isConnected,
  isImageMode,
  modal,
}) => {
  return (
    <div className=" p-2 mt-6 self-center flex flex-col text-white items-center text-center">
      <table>
        <tbody>
          <tr>
            <td>Text temperature:</td>
            <td className="w-12 text-center">{textTemperature}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="text-temperature" name="text-temperature" step="0.01" min="0.2" max="1.2" value={textTemperature} onChange={e => setTextTemperature(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Text topk:</td>
            <td className="w-12 text-center">{textTopk}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="text-topk" name="text-topk" step="1" min="10" max="500" value={textTopk} onChange={e => setTextTopk(parseInt(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Audio temperature:</td>
            <td className="w-12 text-center">{audioTemperature}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-temperature" name="audio-temperature" step="0.01" min="0.2" max="1.2" value={audioTemperature} onChange={e => setAudioTemperature(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Audio topk:</td>
            <td className="w-12 text-center">{audioTopk}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-topk" name="audio-topk" step="1" min="10" max="500" value={audioTopk} onChange={e => setAudioTopk(parseInt(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Padding multiplier:</td>
            <td className="w-12 text-center">{padMult}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-pad-mult" name="audio-pad-mult" step="0.05" min="-4" max="4" value={padMult} onChange={e => setPadMult(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Repeat penalty:</td>
            <td className="w-12 text-center">{repetitionPenalty}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="repetition-penalty" name="repetition-penalty" step="0.01" min="1" max="2" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Repeat penalty last N:</td>
            <td className="w-12 text-center">{repetitionPenaltyContext}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="repetition-penalty-context" name="repetition-penalty-context" step="1" min="0" max="200" value={repetitionPenaltyContext} onChange={e => setRepetitionPenaltyContext(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td>Voice:</td>
            <td className="w-12 text-center text-xs">{voicePrompt || "default"}</td>
            <td className="p-2">
              <select className="bg-gray-800 text-white rounded p-1 text-sm" disabled={isConnected}
                value={voicePrompt} onChange={e => setVoicePrompt(e.target.value)}>
                <option value="">Default</option>
                <optgroup label="Natural Female">
                  <option value="NATF0.safetensors">NATF0</option>
                  <option value="NATF1.pt">NATF1</option>
                  <option value="NATF2.pt">NATF2</option>
                  <option value="NATF3.pt">NATF3</option>
                </optgroup>
                <optgroup label="Natural Male">
                  <option value="NATM0.pt">NATM0</option>
                  <option value="NATM1.pt">NATM1</option>
                  <option value="NATM2.pt">NATM2</option>
                  <option value="NATM3.pt">NATM3</option>
                </optgroup>
                <optgroup label="Varied Female">
                  <option value="VARF0.pt">VARF0</option>
                  <option value="VARF1.pt">VARF1</option>
                  <option value="VARF2.pt">VARF2</option>
                  <option value="VARF3.pt">VARF3</option>
                  <option value="VARF4.pt">VARF4</option>
                </optgroup>
                <optgroup label="Varied Male">
                  <option value="VARM0.pt">VARM0</option>
                  <option value="VARM1.pt">VARM1</option>
                  <option value="VARM2.pt">VARM2</option>
                  <option value="VARM3.pt">VARM3</option>
                  <option value="VARM4.pt">VARM4</option>
                </optgroup>
              </select>
            </td>
          </tr>
          <tr>
            <td>System prompt:</td>
            <td colSpan={2} className="p-2">
              <input className="bg-gray-800 text-white rounded p-1 text-sm w-full" disabled={isConnected}
                type="text" placeholder="e.g. You are a helpful assistant named Jane."
                value={textPrompt} onChange={e => setTextPrompt(e.target.value)} />
            </td>
          </tr>
          {isImageMode &&
            <tr>
              <td>Image max-side (px):</td>
              <td className="w-12 text-center">{imageResolution}</td>
              <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="image-resolution" name="image-resolution" step="16" min="64" max="512" value={imageResolution} onChange={e => setImageResolution(parseFloat(e.target.value))} /></td>
            </tr>
          }
        </tbody>
      </table>
      <div>
        {!isConnected && <Button onClick={resetParams} className="m-2">Reset</Button>}
        {!isConnected && <Button onClick={() => modal?.current?.close()} className="m-2">Validate</Button>}
      </div>
    </div >
  )
};
