/**
 * AniGuardian — The Flow Guardian personality engine.
 * Generates context-aware, personality-driven responses based on flow state analysis.
 * Implements the decision tree logic from the spec + Pomodoro focus timer.
 */
class AniGuardian {
    constructor() {
        this.focusTimer = null;
        this.focusTimeRemaining = 0;
        this.isFocusMode = false;
        this.messageHistory = [];
        this.lastState = null;
        this.consecutiveBadStates = 0;
        this.consecutiveGoodStates = 0;
        this.sessionInsights = { stateChanges: 0, avgQuality: 0, samples: 0 };
    }

    generateResponse(result) {
        const { flowState, vision, audio, nlp, probabilities, confidence, workQuality } = result;
        
        // Track state transitions and dips
        let didDip = false;
        if (this.lastWorkQuality !== undefined) {
            // A dip is a >15% drop in quality OR dropping from good flow to bad state
            if (workQuality < this.lastWorkQuality - 0.15 || (this.lastState >= 3 && flowState <= 2)) {
                didDip = true;
            }
        }
        this.lastWorkQuality = workQuality;

        if (this.lastState !== null && this.lastState !== flowState) {
            this.sessionInsights.stateChanges++;
        }
        this.lastState = flowState;
        this.sessionInsights.samples++;
        this.sessionInsights.avgQuality = 
            ((this.sessionInsights.avgQuality * (this.sessionInsights.samples - 1)) + workQuality) 
            / this.sessionInsights.samples;

        // Track consecutive states
        if (flowState <= 2) { // Bad states
            this.consecutiveBadStates++;
            this.consecutiveGoodStates = 0;
        } else { // Good states
            this.consecutiveGoodStates++;
            this.consecutiveBadStates = 0;
        }

        // Build context for decision tree
        const ctx = this._buildContext(result);
        
        // Run decision tree
        let decision = this._runDecisionTree(ctx);
        
        // DIP OVERRIDE: If the graph just dipped and we know why, explicitly call it out!
        if (didDip && flowState <= 2) {
            let specificDistraction = null;
            if (ctx.hasPhone) {
                specificDistraction = 'your phone';
            } else if (vision?.activeTabTitle && (vision.activeTabCategory === 'distraction' || vision.activeTabCategory === 'news')) {
                // Extract shorter app name from "Video Name - YouTube"
                specificDistraction = vision.activeTabTitle.split(/[-|—]/).pop().trim();
                // If it couldn't find a clean string, just use "this website"
                if (!specificDistraction || specificDistraction.length > 20) specificDistraction = 'this website';
            } else if (vision?.distractions > 0) {
                specificDistraction = 'the background distractions';
            } else if (ctx.hasManyTabs) {
                specificDistraction = 'opening so many tabs';
            }
            
            if (specificDistraction) {
                decision = {
                    state: 'DIP_DETECTED',
                    mood: 'concerned',
                    severity: 'high',
                    reason: 'quality_dip',
                    actions: [`Stop using ${specificDistraction}`, 'Take a deep breath and close it immediately.']
                };
                
                // We'll inject the dynamic text directly into the decision object for the message generator
                ctx.dynamicDipText = `📉 Your flow just dipped! Please stop using **${specificDistraction}** immediately. It's draining your focus.`;
            }
        }
        
        // Generate Ani's message with personality
        const message = this._generateMessage(decision, ctx);
        
        // Determine if we should suggest focus mode
        const suggestFocus = decision.severity === 'high' && !this.isFocusMode && this.consecutiveBadStates >= 2;
        
        const response = {
            message: message.text,
            evidence: message.evidence,
            emoji: message.emoji,
            mood: decision.mood,
            severity: decision.severity,
            suggestFocus,
            focusDuration: 25, // Pomodoro default
            actionItems: decision.actions,
            workQualityProbability: (workQuality * 100).toFixed(0),
            timestamp: Date.now(),
        };

        this.messageHistory.push(response);
        return response;
    }

    /** Build analysis context from all modalities */
    _buildContext(result) {
        const tabCount = result.vision?.tabCount ?? 0;
        const phoneVisible = result.vision?.phoneVisible === 'Yes';
        const distractions = result.vision?.distractions ?? 0;
        const focusRatio = parseFloat(result.vision?.focusRatio) || 50;
        
        const speechClass = result.audio?.speechClass ?? 'Normal';
        const wpm = result.audio?.wpm ?? 130;
        const fluency = parseFloat(result.audio?.fluency) || 70;
        
        const taskType = result.nlp?.taskType ?? 'UNKNOWN';
        const cogDemand = parseFloat(result.nlp?.demand) || 50;
        
        const isComplex = ['DEEP_WORK', 'CREATIVE'].includes(taskType) || cogDemand > 60;
        const isSpeechErratic = ['Erratic', 'Rapid'].includes(speechClass) || wpm > 200 || fluency < 40;
        const isSpeechSlow = speechClass === 'Slow' || wpm < 80;
        const hasManyTabs = tabCount > 10;
        const hasPhone = phoneVisible;
        const hasManyDistractions = distractions >= 3;

        return {
            tabCount, phoneVisible: hasPhone, distractions, focusRatio,
            speechClass, wpm, fluency, taskType, cogDemand,
            isComplex, isSpeechErratic, isSpeechSlow, hasManyTabs,
            hasPhone, hasManyDistractions,
            flowState: result.flowState,
            flowStateName: result.flowStateName,
            confidence: result.confidence,
            workQuality: result.workQuality,
            consecutiveBad: this.consecutiveBadStates,
            consecutiveGood: this.consecutiveGoodStates,
        };
    }

    /** Spec-matching decision tree */
    _runDecisionTree(ctx) {
        // Branch 1 (from spec): Tabs > 10 + Erratic speech + Complex task → Task Switching
        if (ctx.hasManyTabs && ctx.isSpeechErratic && ctx.isComplex) {
            return {
                state: 'TASK_SWITCHING_OVERLOAD',
                mood: 'concerned',
                severity: 'high',
                reason: 'tabs_speech_complex',
                actions: ['Close unnecessary tabs', 'Take a 2-min breathing break', 'Refocus on ONE task'],
            };
        }

        // Branch 2: Phone visible + many distractions → Distracted
        if (ctx.hasPhone && ctx.hasManyDistractions) {
            return {
                state: 'DISTRACTED_PHONE',
                mood: 'worried',
                severity: 'high',
                reason: 'phone_distractions',
                actions: ['Put your phone face-down or in another room', 'Clear your desk', 'Set a focus timer'],
            };
        }

        // Branch 3: Phone visible + low focus
        if (ctx.hasPhone && ctx.focusRatio < 40) {
            return {
                state: 'DISTRACTED_PHONE_MILD',
                mood: 'concerned',
                severity: 'medium',
                reason: 'phone_low_focus',
                actions: ['Move phone out of sight', 'Refocus on your primary task'],
            };
        }

        // Branch 4: Slow labored speech + high demand → Struggling
        if (ctx.isSpeechSlow && ctx.isComplex) {
            return {
                state: 'STRUGGLING',
                mood: 'supportive',
                severity: 'medium',
                reason: 'slow_speech_complex',
                actions: ['Break the task into smaller pieces', 'Take a short walk', 'Try rubber duck debugging'],
            };
        }

        // Branch 5: Many tabs + admin task → Pseudo working
        if (ctx.hasManyTabs && ctx.taskType === 'ADMINISTRATIVE') {
            return {
                state: 'PSEUDO_WORKING',
                mood: 'gentle',
                severity: 'medium',
                reason: 'tabs_admin',
                actions: ['Batch your admin tasks', 'Close tabs you finished with', 'Set a time limit for admin work'],
            };
        }

        // Branch 6: High focus + few distractions + steady speech → Deep flow
        if (ctx.focusRatio > 60 && !ctx.hasPhone && ctx.distractions <= 1 && !ctx.isSpeechErratic && ctx.isComplex) {
            return {
                state: 'DEEP_FLOW',
                mood: 'happy',
                severity: 'none',
                reason: 'everything_aligned',
                actions: [],
            };
        }

        // Branch 7: Good focus, moderate conditions → Soft flow
        if (ctx.focusRatio > 50 && !ctx.hasPhone && ctx.fluency > 50) {
            return {
                state: 'SOFT_FLOW',
                mood: 'pleased',
                severity: 'low',
                reason: 'decent_focus',
                actions: ['Keep going!', 'Consider silencing notifications'],
            };
        }

        // Default: match the ML prediction
        const defaultStates = {
            0: { state: 'PSEUDO_WORKING', mood: 'concerned', severity: 'medium', reason: 'ml_prediction', actions: ['Try a more engaging task', 'Set a clear goal for the next 25 minutes'] },
            1: { state: 'TASK_SWITCHING', mood: 'concerned', severity: 'medium', reason: 'ml_prediction', actions: ['Pick ONE task and commit', 'Use a sticky note for other ideas'] },
            2: { state: 'DISTRACTED', mood: 'gentle', severity: 'medium', reason: 'ml_prediction', actions: ['Remove one distraction', 'Set a micro-goal: focus for just 5 minutes'] },
            3: { state: 'SOFT_FLOW', mood: 'pleased', severity: 'low', reason: 'ml_prediction', actions: ['Good momentum — keep going!'] },
            4: { state: 'DEEP_FLOW', mood: 'happy', severity: 'none', reason: 'ml_prediction', actions: [] },
        };
        return defaultStates[ctx.flowState] || defaultStates[2];
    }

    /** Generate personality-driven Ani message */
    _generateMessage(decision, ctx) {
        const templates = {
            TASK_SWITCHING_OVERLOAD: [
                { emoji: '😟', text: `I see ${ctx.tabCount} tabs open and your speech rate is erratic. My model says you're "Pseudo-Working." I'm suggesting a 25-minute focus sprint so you can actually get into the zone. Focus, okay?` },
                { emoji: '🔍', text: `${ctx.tabCount} tabs, scattered speech, and a complex task — that's a recipe for context-switching burnout. Pick your most important tab, minimize the rest, and let's do a focused sprint.` },
                { emoji: '⚡', text: `Your brain is trying to multitask on a ${ctx.taskType.replace('_', ' ').toLowerCase()} task with ${ctx.tabCount} tabs pulling your attention. My prediction: ${(ctx.workQuality * 100).toFixed(0)}% work quality. Let me help you focus.` },
            ],
            DISTRACTED_PHONE: [
                { emoji: '📱', text: `I noticed your phone is nearby and there are ${ctx.distractions} distractions in view. Your focus ratio is only ${ctx.focusRatio}%. Put the phone in a drawer — out of sight, out of mind!` },
                { emoji: '🚫', text: `Phone detected + ${ctx.distractions} distractions = your brain is in constant interrupt mode. Every glance at your phone costs you 23 minutes of deep focus. Let's fix that.` },
            ],
            DISTRACTED_PHONE_MILD: [
                { emoji: '👀', text: `I can see your phone nearby. Even having it visible reduces your cognitive capacity by ~10%. Try flipping it over or putting it aside.` },
                { emoji: '📵', text: `Your phone is in view and your focus ratio dipped to ${ctx.focusRatio}%. A small change — just moving it — can make a big difference.` },
            ],
            STRUGGLING: [
                { emoji: '💪', text: `Your speech pattern suggests you're working hard on something challenging — that's good! But you seem to be struggling. Try breaking this ${ctx.taskType.replace('_', ' ').toLowerCase()} task into smaller sub-tasks.` },
                { emoji: '🧩', text: `I'm picking up slow, deliberate speech on a high-demand task. You might be overthinking it. Try writing down just the NEXT small step, not the whole solution.` },
            ],
            PSEUDO_WORKING: [
                { emoji: '🤔', text: `Hmm, ${ctx.tabCount} tabs open but low cognitive engagement detected. Are you actually making progress, or just "busy"? Try setting a concrete micro-goal for the next 10 minutes.` },
                { emoji: '💡', text: `My sensors suggest you're in "busy but not productive" mode. That's okay — it happens! Here's a trick: close everything, write down ONE thing you want to accomplish, then reopen only what you need.` },
            ],
            TASK_SWITCHING: [
                { emoji: '🔄', text: `I'm detecting rapid context-switching patterns. Every switch costs you 5-15 minutes of ramp-up time. Pick your top priority and give it an uninterrupted 25-minute block.` },
                { emoji: '🎯', text: `You're bouncing between contexts. My model gives you ${(ctx.workQuality * 100).toFixed(0)}% work quality right now. Let's boost that — commit to one task for the next Pomodoro.` },
            ],
            DISTRACTED: [
                { emoji: '🌊', text: `Focus is drifting. ${ctx.hasPhone ? 'I see your phone. ' : ''}${ctx.distractions > 0 ? `I noticed ${ctx.distractions} visual distraction(s). ` : ''}${ctx.tabCount > 10 ? `You have ${ctx.tabCount} tabs pulling your attention. ` : ''}Try the "5-minute rule" — commit to focused work for just 5 minutes without these.` },
                { emoji: '🧘', text: `Your attention seems scattered! ${ctx.hasPhone ? 'Your phone is right there. ' : ''}${ctx.focusRatio < 50 ? `Your visual focus is only ${(ctx.focusRatio).toFixed(0)}%. ` : ''}Take a deep breath, minimize extras, and ask: "What's the ONE thing I need to do right now?"` },
                { emoji: '⚠️', text: `I'm picking up a distracted state. Reason: ${[ctx.hasPhone ? 'Phone detected' : '', ctx.distractions > 0 ? `${ctx.distractions} external distractions` : '', ctx.tabCount > 10 ? 'Too many tabs' : '', ctx.isSpeechErratic ? 'Erratic speech' : ''].filter(Boolean).join(', ') || 'General scattered focus'}. Let's remove the noise and jump back in.` }
            ],
            SOFT_FLOW: [
                { emoji: '🟢', text: `You're in a good rhythm! Focus ratio is solid at ${ctx.focusRatio}% and your speech pattern is steady. Keep this up — you're building toward deep flow.` },
                { emoji: '✨', text: `Nice work! You're in Soft Flow right now. Work quality probability: ${(ctx.workQuality * 100).toFixed(0)}%. Stay the course and you might hit Deep Flow soon.` },
                { emoji: '🚀', text: `Good momentum detected — ${ctx.wpm} WPM with ${ctx.fluency}% fluency. You're focused. I'll keep quiet unless something changes. Keep going!` },
            ],
            DEEP_FLOW: [
                { emoji: '🟣', text: `🔥 You're in DEEP FLOW! Everything's aligned — clear focus, minimal distractions, steady engagement. I'm going silent to protect this. Don't stop!` },
                { emoji: '💜', text: `Peak performance detected! Work quality: ${(ctx.workQuality * 100).toFixed(0)}%. This is the zone where your best work happens. I won't disturb you.` },
                { emoji: '🧠', text: `Deep Flow state confirmed. ${ctx.tabCount <= 5 ? 'Clean workspace, ' : ''}${ctx.fluency > 60 ? 'steady voice, ' : ''}high cognitive demand — you're crushing it. I'll stay quiet.` },
            ],
            DIP_DETECTED: [
                { emoji: '📉', text: ctx.dynamicDipText }
            ]
        };

        const stateTemplates = templates[decision.state] || templates['DISTRACTED'];
        
        // Add urgency prefix for consecutive bad states
        let chosen = stateTemplates[Math.floor(Math.random() * stateTemplates.length)];
        
        if (this.consecutiveBadStates >= 3 && decision.severity !== 'none') {
            chosen = { 
                emoji: '🚨', 
                text: `Hey, this is the ${this.consecutiveBadStates}th check in a row where you're not in flow. ${chosen.text}` 
            };
        }

        // Add encouragement for improvement
        if (this.consecutiveGoodStates === 1 && this.sessionInsights.stateChanges > 0) {
            chosen = {
                emoji: '🎉',
                text: `Great improvement! You just moved into a better flow state. ${chosen.text}`,
            };
        }

        return chosen;
    }

    /** Start Pomodoro focus timer */
    startFocusMode(durationMinutes = 25) {
        this.isFocusMode = true;
        this.focusTimeRemaining = durationMinutes * 60;
        
        this.focusTimer = setInterval(() => {
            this.focusTimeRemaining--;
            if (this.focusTimeRemaining <= 0) {
                this.endFocusMode();
            }
        }, 1000);

        return {
            emoji: '🔇',
            message: `Focus mode activated! I'm muting distractions for ${durationMinutes} minutes. You've got this — see you on the other side! 💪`,
            duration: durationMinutes,
        };
    }

    /** End Pomodoro focus timer */
    endFocusMode() {
        this.isFocusMode = false;
        this.focusTimeRemaining = 0;
        if (this.focusTimer) {
            clearInterval(this.focusTimer);
            this.focusTimer = null;
        }
        return {
            emoji: '🔔',
            message: `Focus sprint complete! Great work. Take a 5-minute break — stretch, hydrate, look away from the screen. You've earned it.`,
        };
    }

    /** Get formatted time remaining */
    getFocusTimeFormatted() {
        const mins = Math.floor(this.focusTimeRemaining / 60);
        const secs = this.focusTimeRemaining % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    /** Get session summary */
    getSessionSummary() {
        if (this.sessionInsights.samples === 0) return null;
        const quality = (this.sessionInsights.avgQuality * 100).toFixed(0);
        return {
            emoji: quality > 60 ? '🏆' : quality > 40 ? '📊' : '🎯',
            message: `Session recap: ${this.sessionInsights.samples} analyses, ${this.sessionInsights.stateChanges} state changes, average work quality ${quality}%. ${quality > 60 ? 'Great session!' : 'Room to improve next time.'}`,
            avgQuality: this.sessionInsights.avgQuality,
            stateChanges: this.sessionInsights.stateChanges,
            totalSamples: this.sessionInsights.samples,
        };
    }

    /** Reset for new session */
    reset() {
        this.lastState = null;
        this.consecutiveBadStates = 0;
        this.consecutiveGoodStates = 0;
        this.sessionInsights = { stateChanges: 0, avgQuality: 0, samples: 0 };
        this.messageHistory = [];
        if (this.focusTimer) {
            clearInterval(this.focusTimer);
            this.focusTimer = null;
        }
        this.isFocusMode = false;
        this.focusTimeRemaining = 0;
    }
}

window.AniGuardian = AniGuardian;
