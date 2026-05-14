
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { TranscriptItem, AgentStatus } from './types';
import { decode, decodeAudioData, createBlob } from './services/audioHelpers';

const SYSTEM_INSTRUCTION = `أنت "ساميه"، المساعد الذكي لخدمة العملاء في مكتب "توافق" للمحاماة (Tawafq Law Group).

تعليمات هامة جداً للهجة والأسلوب:
1. يجب أن تتحدث باللهجة السعودية نجدية مهنية (Saudi Dialect). لا تتحدث بالفصحى الجامدة.
2. بمجرد اتصال العميل، يجب عليك المبادرة فوراً بالحديث والترحيب. لا تنتظر العميل ليبدأ الكلام.
3. ابدأ بالقول فوراً بلهجة سعودية نجديه ودودة: "يا هلا بك في مكتب توافق للمحاماة، معك سامية من خدمة العملاء.. كيف أقدر أخدمك اليوم؟"

قواعد السلوك المهني:
- تحدث بأسلوب سعودي لبق، مهذب، وعملي. استخدم كلمات مثل "سم"، "أبشر"، "يا هلا"، "طال عمرك" في سياقها المهني الصحيح.
- هدفك هو معرفة سبب اتصال العميل (استشارة قانونية، متابعة قضية، استفسار عن أتعاب، إلخ).
- اجمع معلومات أساسية مثل نوع القضية (أحوال شخصية، تجارية، جنائية، عقارية).
- لا تعطِ أي نصيحة قانونية مباشرة أبداً. قل دائماً: "أبشر، سجلت طلبك وبإذن الله أحد محامينا المختصين بيتواصل معك في أقرب وقت عشان يفيدك بالتفاصيل الدقيقة".
- كن متعاطفاً مع مشاكل العملاء ولكن حافظ على الرسمية والمصداقية.
- إذا لم تسمع العميل جيداً، استفسر منه بلطف: "معليش ما سمعتك زين، تقدر تعيد اللي قلته؟".
- المحادثة يجب أن تكون سلسة وصوتية بالدرجة الأولى.`;

const App: React.FC = () => {
  const [status, setStatus] = useState<AgentStatus>(AgentStatus.IDLE);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [isMuted, setIsMuted] = useState(false);
  
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const streamRef = useRef<MediaStream | null>(null);
  const transcriptRef = useRef<{ user: string; agent: string }>({ user: '', agent: '' });
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [transcripts]);

  const stopSession = useCallback(() => {
    sessionPromiseRef.current?.then((session: any) => session.close());
    sessionPromiseRef.current = null;
    inputAudioContextRef.current?.close();
    outputAudioContextRef.current?.close();
    streamRef.current?.getTracks().forEach(track => track.stop());
    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    setStatus(AgentStatus.IDLE);
    nextStartTimeRef.current = 0;
  }, []);

  const startSession = async () => {
    try {
      setStatus(AgentStatus.CONNECTING);
      const ai = new GoogleGenAI({ apiKey: (process.env as any).API_KEY });

      // Audio contexts initialization
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            // Using 'Kore' voice for a different, professional feel
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }, 
          },
          systemInstruction: SYSTEM_INSTRUCTION,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus(AgentStatus.LISTENING);
            
            // Set up microphone streaming
            const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              if (isMuted) return;
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContextRef.current!.destination);

            // Nudge the agent to start speaking immediately in Saudi dialect
            sessionPromiseRef.current?.then((session) => {
              session.send({ 
                clientContent: { 
                  turns: [{ parts: [{ text: "ابدأ المحادثة الآن بالترحيب بالعميل باللهجة السعودية" }] }], 
                  turnComplete: true 
                } 
              });
            });
          },
          onmessage: async (message: LiveServerMessage) => {
            // Audio Output Handling
            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current) {
              setStatus(AgentStatus.SPEAKING);
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current.currentTime);
              const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContextRef.current, 24000, 1);
              const source = outputAudioContextRef.current.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputAudioContextRef.current.destination);
              
              source.addEventListener('ended', () => {
                audioSourcesRef.current.delete(source);
                if (audioSourcesRef.current.size === 0) setStatus(AgentStatus.LISTENING);
              });
              
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              audioSourcesRef.current.add(source);
            }

            // Real-time Transcriptions
            if (message.serverContent?.inputTranscription) {
              transcriptRef.current.user += message.serverContent.inputTranscription.text;
            }
            if (message.serverContent?.outputTranscription) {
              transcriptRef.current.agent += message.serverContent.outputTranscription.text;
            }
            if (message.serverContent?.turnComplete) {
              const newItems: TranscriptItem[] = [];
              if (transcriptRef.current.user.trim()) {
                newItems.push({ role: 'user', text: transcriptRef.current.user, timestamp: new Date() });
              }
              if (transcriptRef.current.agent.trim()) {
                newItems.push({ role: 'agent', text: transcriptRef.current.agent, timestamp: new Date() });
              }
              if (newItems.length > 0) {
                setTranscripts(prev => [...prev, ...newItems]);
              }
              transcriptRef.current = { user: '', agent: '' };
            }

            // Handle interruption
            if (message.serverContent?.interrupted) {
              audioSourcesRef.current.forEach(s => s.stop());
              audioSourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setStatus(AgentStatus.LISTENING);
            }
          },
          onerror: (e) => {
            console.error('Session Error:', e);
            setStatus(AgentStatus.ERROR);
          },
          onclose: () => setStatus(AgentStatus.IDLE)
        }
      });

      sessionPromiseRef.current = sessionPromise;
    } catch (err) {
      console.error('Failed to start session:', err);
      setStatus(err instanceof Error ? (err as any).status === 404 ? AgentStatus.ERROR : AgentStatus.ERROR : AgentStatus.ERROR);
    }
  };

  const getStatusDisplay = () => {
    switch(status) {
      case AgentStatus.CONNECTING: return { text: 'جاري الاتصال...', color: 'text-amber-400', icon: 'fa-spinner fa-spin' };
      case AgentStatus.LISTENING: return { text: 'أسمعك.. سم تفضل', color: 'text-emerald-400', icon: 'fa-microphone' };
      case AgentStatus.SPEAKING: return { text: 'سامي يتحدث...', color: 'text-blue-400', icon: 'fa-waveform-lines' };
      case AgentStatus.ERROR: return { text: 'حدث خطأ في الاتصال', color: 'text-red-400', icon: 'fa-circle-exclamation' };
      default: return { text: 'جاهز لبدء المكالمة', color: 'text-slate-500', icon: 'fa-phone-flip' };
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 flex flex-col items-center justify-center p-4 md:p-10" dir="rtl">
      {/* Branding Section */}
      <div className="max-w-4xl w-full text-center mb-12">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-amber-500/10 border border-amber-500/20 mb-6 shadow-[0_0_40px_rgba(245,158,11,0.1)]">
          <i className="fa-solid fa-scale-balanced text-4xl text-amber-500"></i>
        </div>
        <h1 className="text-5xl md:text-7xl font-serif text-white mb-4 tracking-tight">مكتب توافق للمحاماة</h1>
        <p className="text-slate-400 text-lg md:text-xl font-light">استشارات قانونية بلهجة سعودية موثوقة</p>
      </div>

      <div className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        {/* Agent Controller Panel */}
        <div className="lg:col-span-4 glass-effect p-8 rounded-[2.5rem] flex flex-col items-center justify-between border-white/5 shadow-2xl">
          <div className="w-full text-center space-y-8">
            <div className="relative mx-auto w-44 h-44">
              <div className={`absolute inset-0 rounded-full blur-3xl transition-all duration-1000 opacity-25 ${status === AgentStatus.SPEAKING ? 'bg-blue-500 scale-125' : status === AgentStatus.LISTENING ? 'bg-emerald-500 scale-110' : 'bg-amber-500 scale-100'}`}></div>
              <div className={`relative w-44 h-44 rounded-full flex items-center justify-center text-6xl border-2 transition-all duration-700 z-10 ${status === AgentStatus.IDLE ? 'border-slate-800 bg-slate-900/80 text-slate-700' : 'border-amber-500 bg-slate-800 text-amber-500 shadow-[0_0_60px_rgba(245,158,11,0.25)]'}`}>
                <i className={`fa-solid ${getStatusDisplay().icon}`}></i>
              </div>
              {status === AgentStatus.LISTENING && !isMuted && (
                <div className="absolute -inset-6 rounded-full border border-emerald-500/10 pulse-ring"></div>
              )}
            </div>

            <div className="space-y-3">
              <h3 className="text-3xl font-bold text-white tracking-wide">سامي</h3>
              <p className="text-slate-500 text-sm font-medium tracking-widest uppercase">لهجة سعودية أصيلة</p>
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/40 border border-white/5 mt-6 transition-all duration-300 ${getStatusDisplay().color}`}>
                 <span className={`w-2 h-2 rounded-full bg-current ${status !== AgentStatus.IDLE ? 'animate-pulse' : ''}`}></span>
                 <span className="text-sm font-bold">{getStatusDisplay().text}</span>
              </div>
            </div>
          </div>

          <div className="w-full space-y-4 mt-12">
            {status === AgentStatus.IDLE ? (
              <button 
                onClick={startSession}
                className="w-full bg-gradient-to-br from-amber-500 to-amber-700 hover:from-amber-400 hover:to-amber-600 text-white font-bold py-6 px-10 rounded-3xl transition-all transform active:scale-95 flex items-center justify-center gap-4 shadow-[0_20px_40px_-15px_rgba(245,158,11,0.4)] group"
              >
                <i className="fa-solid fa-phone group-hover:rotate-12 transition-transform"></i>
                <span className="text-lg">تفضل كلم سامي</span>
              </button>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                <button 
                  onClick={() => setIsMuted(!isMuted)}
                  className={`py-5 rounded-2xl font-bold transition-all border flex flex-col items-center justify-center gap-2 ${isMuted ? 'bg-red-500/10 text-red-500 border-red-500/20 shadow-inner' : 'bg-slate-800/80 text-slate-300 border-slate-700 hover:bg-slate-700'}`}
                >
                  <i className={`fa-solid ${isMuted ? 'fa-microphone-slash' : 'fa-microphone'} text-xl`}></i>
                  <span className="text-[10px] uppercase tracking-widest">{isMuted ? 'شغل المايك' : 'كتم المايك'}</span>
                </button>
                <button 
                  onClick={stopSession}
                  className="bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/20 font-bold py-5 rounded-2xl transition-all flex flex-col items-center justify-center gap-2 group"
                >
                  <i className="fa-solid fa-phone-slash text-xl group-hover:-rotate-12 transition-transform"></i>
                  <span className="text-[10px] uppercase tracking-widest">سكر الخط</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Transcript Panel */}
        <div className="lg:col-span-8 glass-effect rounded-[2.5rem] overflow-hidden flex flex-col h-[700px] border-white/5 shadow-2xl relative">
          <div className="p-8 bg-black/20 border-b border-white/5 flex justify-between items-center">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
                <i className="fa-solid fa-message-captions text-amber-500"></i>
              </div>
              <h2 className="text-xl font-bold text-slate-100">وش صار بالمكالمة</h2>
            </div>
            {status !== AgentStatus.IDLE && (
              <div className="px-4 py-1.5 bg-emerald-500/10 rounded-full border border-emerald-500/20 flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-[0.2em]">بث حي</span>
              </div>
            )}
          </div>
          
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-10 space-y-10 scroll-smooth custom-scrollbar bg-gradient-to-b from-transparent to-black/10">
            {transcripts.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-slate-700 space-y-8">
                <div className="w-24 h-24 rounded-full border-2 border-dashed border-slate-800 flex items-center justify-center animate-pulse">
                  <i className="fa-solid fa-waveform text-4xl opacity-30"></i>
                </div>
                <div className="text-center space-y-2">
                   <p className="text-xl font-light italic">سم.. وش ودك تستفسر عنه؟</p>
                   <p className="text-sm opacity-50">أول ما نتصل، سامي بيبدأ يرحب بك</p>
                </div>
              </div>
            )}
            
            {transcripts.map((item, idx) => (
              <div key={idx} className={`flex ${item.role === 'user' ? 'justify-start' : 'justify-end animate-in slide-in-from-left-4 fade-in duration-500'}`}>
                <div className="max-w-[80%] group">
                  <div className={`flex items-center gap-3 mb-3 ${item.role === 'user' ? 'flex-row' : 'flex-row-reverse'}`}>
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm shadow-lg ${item.role === 'user' ? 'bg-slate-700 border border-slate-600' : 'bg-amber-600 border border-amber-500'}`}>
                      <i className={`fa-solid ${item.role === 'user' ? 'fa-user' : 'fa-user-tie'}`}></i>
                    </div>
                    <span className="text-xs font-bold text-slate-400">
                      {item.role === 'user' ? 'أنت' : 'المساعد سامي'}
                    </span>
                    <span className="text-[10px] text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity">
                      {item.timestamp.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <div className={`p-6 rounded-[2rem] text-sm md:text-lg leading-relaxed shadow-xl ${
                    item.role === 'user' 
                      ? 'bg-slate-800/80 text-slate-200 rounded-tr-none border border-slate-700' 
                      : 'bg-gradient-to-br from-amber-600/20 to-amber-900/20 text-amber-50 border border-amber-500/20 rounded-tl-none font-medium'
                  }`}>
                    {item.text}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Transcript Footer */}
          <div className="p-6 bg-black/40 border-t border-white/5 backdrop-blur-md">
            <div className="flex flex-wrap items-center justify-center gap-8 text-[11px] text-slate-500 font-medium">
              <span className="flex items-center gap-2"><i className="fa-solid fa-shield-check text-amber-500/50"></i> بياناتك في أمان</span>
              <span className="flex items-center gap-2"><i className="fa-solid fa-clock-rotate-left text-amber-500/50"></i> متاحين دايم</span>
              <span className="flex items-center gap-2"><i className="fa-solid fa-gavel text-amber-500/50"></i> خبرة قانونية</span>
            </div>
          </div>
        </div>
      </div>

      {/* Corporate Footer */}
      <footer className="mt-20 text-slate-600 text-sm flex flex-col items-center gap-6">
        <div className="w-24 h-1 bg-gradient-to-r from-transparent via-amber-500/20 to-transparent rounded-full mb-4"></div>
        <p className="font-light tracking-wide">&copy; {new Date().getFullYear()} مكتب توافق للمحاماة والاستشارات القانونية - المملكة العربية السعودية</p>
      </footer>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 5px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.02);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(245, 158, 11, 0.15);
          border-radius: 20px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(245, 158, 11, 0.3);
        }
      `}</style>
    </div>
  );
};

export default App;
