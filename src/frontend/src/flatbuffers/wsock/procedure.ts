// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { FileType } from './file-type';
import { Parameter, ParameterT } from '../wsock/parameter.js';
import { ProcedureState } from './procedure-state';


export class Procedure implements flatbuffers.IUnpackableObject<ProcedureT> {
  bb: flatbuffers.ByteBuffer|null = null;
  bb_pos = 0;
  __init(i:number, bb:flatbuffers.ByteBuffer):Procedure {
  this.bb_pos = i;
  this.bb = bb;
  return this;
}

static getRootAsProcedure(bb:flatbuffers.ByteBuffer, obj?:Procedure):Procedure {
  return (obj || new Procedure()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
}

static getSizePrefixedRootAsProcedure(bb:flatbuffers.ByteBuffer, obj?:Procedure):Procedure {
  bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
  return (obj || new Procedure()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
}

name():string|null
name(optionalEncoding:flatbuffers.Encoding):string|Uint8Array|null
name(optionalEncoding?:any):string|Uint8Array|null {
  const offset = this.bb!.__offset(this.bb_pos, 4);
  return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
}

description():string|null
description(optionalEncoding:flatbuffers.Encoding):string|Uint8Array|null
description(optionalEncoding?:any):string|Uint8Array|null {
  const offset = this.bb!.__offset(this.bb_pos, 6);
  return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
}

scriptFileLocation():string|null
scriptFileLocation(optionalEncoding:flatbuffers.Encoding):string|Uint8Array|null
scriptFileLocation(optionalEncoding?:any):string|Uint8Array|null {
  const offset = this.bb!.__offset(this.bb_pos, 8);
  return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
}

parameters(index: number, obj?:Parameter):Parameter|null {
  const offset = this.bb!.__offset(this.bb_pos, 10);
  return offset ? (obj || new Parameter()).__init(this.bb!.__indirect(this.bb!.__vector(this.bb_pos + offset) + index * 4), this.bb!) : null;
}

parametersLength():number {
  const offset = this.bb!.__offset(this.bb_pos, 10);
  return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
}

inputFileVar():string|null
inputFileVar(optionalEncoding:flatbuffers.Encoding):string|Uint8Array|null
inputFileVar(optionalEncoding?:any):string|Uint8Array|null {
  const offset = this.bb!.__offset(this.bb_pos, 12);
  return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
}

inputFileType():FileType {
  const offset = this.bb!.__offset(this.bb_pos, 14);
  return offset ? this.bb!.readInt8(this.bb_pos + offset) : FileType.IFC;
}

exportFileType():FileType {
  const offset = this.bb!.__offset(this.bb_pos, 16);
  return offset ? this.bb!.readInt8(this.bb_pos + offset) : FileType.IFC;
}

exportFileVar():string|null
exportFileVar(optionalEncoding:flatbuffers.Encoding):string|Uint8Array|null
exportFileVar(optionalEncoding?:any):string|Uint8Array|null {
  const offset = this.bb!.__offset(this.bb_pos, 18);
  return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
}

state():ProcedureState {
  const offset = this.bb!.__offset(this.bb_pos, 20);
  return offset ? this.bb!.readInt8(this.bb_pos + offset) : ProcedureState.IDLE;
}

isComponent():boolean {
  const offset = this.bb!.__offset(this.bb_pos, 22);
  return offset ? !!this.bb!.readInt8(this.bb_pos + offset) : false;
}

static startProcedure(builder:flatbuffers.Builder) {
  builder.startObject(10);
}

static addName(builder:flatbuffers.Builder, nameOffset:flatbuffers.Offset) {
  builder.addFieldOffset(0, nameOffset, 0);
}

static addDescription(builder:flatbuffers.Builder, descriptionOffset:flatbuffers.Offset) {
  builder.addFieldOffset(1, descriptionOffset, 0);
}

static addScriptFileLocation(builder:flatbuffers.Builder, scriptFileLocationOffset:flatbuffers.Offset) {
  builder.addFieldOffset(2, scriptFileLocationOffset, 0);
}

static addParameters(builder:flatbuffers.Builder, parametersOffset:flatbuffers.Offset) {
  builder.addFieldOffset(3, parametersOffset, 0);
}

static createParametersVector(builder:flatbuffers.Builder, data:flatbuffers.Offset[]):flatbuffers.Offset {
  builder.startVector(4, data.length, 4);
  for (let i = data.length - 1; i >= 0; i--) {
    builder.addOffset(data[i]!);
  }
  return builder.endVector();
}

static startParametersVector(builder:flatbuffers.Builder, numElems:number) {
  builder.startVector(4, numElems, 4);
}

static addInputFileVar(builder:flatbuffers.Builder, inputFileVarOffset:flatbuffers.Offset) {
  builder.addFieldOffset(4, inputFileVarOffset, 0);
}

static addInputFileType(builder:flatbuffers.Builder, inputFileType:FileType) {
  builder.addFieldInt8(5, inputFileType, FileType.IFC);
}

static addExportFileType(builder:flatbuffers.Builder, exportFileType:FileType) {
  builder.addFieldInt8(6, exportFileType, FileType.IFC);
}

static addExportFileVar(builder:flatbuffers.Builder, exportFileVarOffset:flatbuffers.Offset) {
  builder.addFieldOffset(7, exportFileVarOffset, 0);
}

static addState(builder:flatbuffers.Builder, state:ProcedureState) {
  builder.addFieldInt8(8, state, ProcedureState.IDLE);
}

static addIsComponent(builder:flatbuffers.Builder, isComponent:boolean) {
  builder.addFieldInt8(9, +isComponent, +false);
}

static endProcedure(builder:flatbuffers.Builder):flatbuffers.Offset {
  const offset = builder.endObject();
  return offset;
}

static createProcedure(builder:flatbuffers.Builder, nameOffset:flatbuffers.Offset, descriptionOffset:flatbuffers.Offset, scriptFileLocationOffset:flatbuffers.Offset, parametersOffset:flatbuffers.Offset, inputFileVarOffset:flatbuffers.Offset, inputFileType:FileType, exportFileType:FileType, exportFileVarOffset:flatbuffers.Offset, state:ProcedureState, isComponent:boolean):flatbuffers.Offset {
  Procedure.startProcedure(builder);
  Procedure.addName(builder, nameOffset);
  Procedure.addDescription(builder, descriptionOffset);
  Procedure.addScriptFileLocation(builder, scriptFileLocationOffset);
  Procedure.addParameters(builder, parametersOffset);
  Procedure.addInputFileVar(builder, inputFileVarOffset);
  Procedure.addInputFileType(builder, inputFileType);
  Procedure.addExportFileType(builder, exportFileType);
  Procedure.addExportFileVar(builder, exportFileVarOffset);
  Procedure.addState(builder, state);
  Procedure.addIsComponent(builder, isComponent);
  return Procedure.endProcedure(builder);
}

unpack(): ProcedureT {
  return new ProcedureT(
    this.name(),
    this.description(),
    this.scriptFileLocation(),
    this.bb!.createObjList<Parameter, ParameterT>(this.parameters.bind(this), this.parametersLength()),
    this.inputFileVar(),
    this.inputFileType(),
    this.exportFileType(),
    this.exportFileVar(),
    this.state(),
    this.isComponent()
  );
}


unpackTo(_o: ProcedureT): void {
  _o.name = this.name();
  _o.description = this.description();
  _o.scriptFileLocation = this.scriptFileLocation();
  _o.parameters = this.bb!.createObjList<Parameter, ParameterT>(this.parameters.bind(this), this.parametersLength());
  _o.inputFileVar = this.inputFileVar();
  _o.inputFileType = this.inputFileType();
  _o.exportFileType = this.exportFileType();
  _o.exportFileVar = this.exportFileVar();
  _o.state = this.state();
  _o.isComponent = this.isComponent();
}
}

export class ProcedureT implements flatbuffers.IGeneratedObject {
constructor(
  public name: string|Uint8Array|null = null,
  public description: string|Uint8Array|null = null,
  public scriptFileLocation: string|Uint8Array|null = null,
  public parameters: (ParameterT)[] = [],
  public inputFileVar: string|Uint8Array|null = null,
  public inputFileType: FileType = FileType.IFC,
  public exportFileType: FileType = FileType.IFC,
  public exportFileVar: string|Uint8Array|null = null,
  public state: ProcedureState = ProcedureState.IDLE,
  public isComponent: boolean = false
){}


pack(builder:flatbuffers.Builder): flatbuffers.Offset {
  const name = (this.name !== null ? builder.createString(this.name!) : 0);
  const description = (this.description !== null ? builder.createString(this.description!) : 0);
  const scriptFileLocation = (this.scriptFileLocation !== null ? builder.createString(this.scriptFileLocation!) : 0);
  const parameters = Procedure.createParametersVector(builder, builder.createObjectOffsetList(this.parameters));
  const inputFileVar = (this.inputFileVar !== null ? builder.createString(this.inputFileVar!) : 0);
  const exportFileVar = (this.exportFileVar !== null ? builder.createString(this.exportFileVar!) : 0);

  return Procedure.createProcedure(builder,
    name,
    description,
    scriptFileLocation,
    parameters,
    inputFileVar,
    this.inputFileType,
    this.exportFileType,
    exportFileVar,
    this.state,
    this.isComponent
  );
}
}
