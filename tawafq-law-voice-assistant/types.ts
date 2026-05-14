
export interface TranscriptItem {
  role: 'user' | 'agent';
  text: string;
  timestamp: Date;
}

export enum AgentStatus {
  IDLE = 'IDLE',
  CONNECTING = 'CONNECTING',
  LISTENING = 'LISTENING',
  SPEAKING = 'SPEAKING',
  ERROR = 'ERROR'
}
