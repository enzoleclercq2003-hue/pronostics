/* Bande-son de la pub, entièrement synthétisée dans le navigateur (Web Audio) : aucun fichier audio,
   donc aucun droit à acquérir. Effets sobres : souffle du tir, impact sourd, scintillement, accord
   de signature, pulsations et tics sur les coupes, nappe discrète. Expose window.Snd. */
(function () {
  let ctx = null, master, send, noise, pad = null;
  const now = () => ctx.currentTime;
  function init() {
    if (ctx) return; const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    ctx = new AC(); master = ctx.createGain(); master.gain.value = .85;
    const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -14; comp.ratio.value = 3; master.connect(comp); comp.connect(ctx.destination);
    // espace : un écho filtré en boucle fait office de réverbération légère
    send = ctx.createGain(); const d = ctx.createDelay(1), lp = ctx.createBiquadFilter(), fb = ctx.createGain(); d.delayTime.value = .21; lp.type = 'lowpass'; lp.frequency.value = 2400; fb.gain.value = .36;
    send.connect(d); d.connect(lp); lp.connect(fb); fb.connect(d); lp.connect(master);
    noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate); const ch = noise.getChannelData(0); for (let i = 0; i < ch.length; i++) ch[i] = Math.random() * 2 - 1;
  }
  function out(g, wet) { g.connect(master); if (wet) { const w = ctx.createGain(); w.gain.value = wet; g.connect(w); w.connect(send); } }
  function tone(type, f1, f2, t, dur, vol, wet, attack) {
    const o = ctx.createOscillator(), g = ctx.createGain(); o.type = type; o.frequency.setValueAtTime(f1, t); if (f2 !== f1) o.frequency.exponentialRampToValueAtTime(Math.max(1, f2), t + dur);
    g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(vol, t + (attack || .005)); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); out(g, wet); o.start(t); o.stop(t + dur + .05);
  }
  function hiss(t, dur, vol, type, f1, f2, q, wet, shape) {
    const s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain(); s.buffer = noise; s.loop = true; f.type = type; f.Q.value = q || .8;
    f.frequency.setValueAtTime(f1, t); f.frequency.exponentialRampToValueAtTime(f2, t + dur);
    g.gain.setValueAtTime(0.0001, t);
    if (shape === 'rise') { g.gain.exponentialRampToValueAtTime(vol, t + dur * .92); g.gain.linearRampToValueAtTime(0.0001, t + dur); }
    else { g.gain.linearRampToValueAtTime(vol, t + .008); g.gain.exponentialRampToValueAtTime(0.0001, t + dur); }
    s.connect(f); f.connect(g); out(g, wet); s.start(t); s.stop(t + dur + .05);
  }
  const N = {Fs2: 92.5, Cs3: 138.6, Fs3: 185, Gs3: 207.7, Cs4: 277.2, Ds4: 311.1, Fs4: 370, Gs4: 415.3, As4: 466.2, B4: 493.9, Cs5: 554.4, Ds5: 622.3, Fs5: 740, Gs5: 830.6, As5: 932.3, Cs6: 1108.7, Ds6: 1244.5, Fs6: 1480, Gs6: 1661.2, As6: 1864.7, Cs7: 2217.5};
  const pluck = (f, t, vol = .22) => { tone('triangle', f, f, t, .5, vol, .35); tone('sine', f * 2, f * 2, t, .28, vol * .35, .3); };
  const tick = (t, f = 1900, vol = .1) => { tone('sine', f, f * .6, t, .035, vol); hiss(t, .02, vol * .5, 'highpass', 5000, 7000, .7); };
  const thump = (t, f = 62, vol = .75) => tone('sine', f * 2.2, f, t, .2, vol);
  const C = {
    whoosh(t) { hiss(t, .84, .5, 'bandpass', 180, 3600, 1.1, .1, 'rise'); tone('sawtooth', 70, 360, t, .82, .05, 0, .5); tone('sine', 220, 1500, t + .3, .52, .05, .2, .3); },
    impact(t) { tone('sine', 150, 34, t, .9, 1, .05); hiss(t, .09, .7, 'highpass', 900, 3000, .7); hiss(t, .45, .5, 'lowpass', 1400, 180, .7, .25); tone('square', 60, 30, t, .12, .2); },
    shimmer(t) { [N.Cs6, N.Fs6, N.Gs6, N.As6, N.Cs7, N.Fs6, N.As6].forEach((f, i) => tone('sine', f, f, t + i * .07, .7, .07, .6)); hiss(t, 1.1, .05, 'highpass', 6000, 9000, .7, .4); },
    chime(t) { [N.Fs3, N.Cs4, N.Fs4, N.As4, N.Cs5, N.Fs5].forEach((f, i) => { tone('sine', f, f, t + i * .012, 3, .2 / (1 + i * .25), .4, .02); tone('triangle', f * 1.003, f * 1.003, t, 2.2, .05, .3, .03); }); tone('sine', N.Cs6, N.Cs6, t + .05, 1.6, .05, .6); },
    swish(t) { hiss(t, .26, .16, 'bandpass', 600, 4200, 1.4, .15); },
    hook(t) { [0, .18, .42].forEach((d, i) => { thump(t + d, 58 + i * 6, .55); tick(t + d, 1500 + i * 300); }); },
    num(t, f) { thump(t, 64, .7); pluck(f, t, .26); },
    cut(t) { thump(t, 58, .5); tick(t, 2200, .09); },
    boom(t) { tone('sine', 130, 38, t, .8, .9, .08); hiss(t, .3, .25, 'lowpass', 1200, 200, .7, .2); [N.Fs3, N.Cs4, N.Fs4].forEach(f => tone('triangle', f, f, t, 1, .12, .4, .01)); },
    deal(t) { [0, .22, .44].forEach((d, i) => { hiss(t + d, .2, .14, 'bandpass', 900, 3200, 1.5, .1); tick(t + d + .17, 1300 + i * 250, .08); }); [N.Ds5, N.Fs5, N.As5].forEach((f, i) => pluck(f, t + .95 + i * .12, .12)); },
    roll(t) { const n = 26; for (let i = 0; i < n; i++) { const k = i / n; tick(t + 1.1 * (1 - Math.pow(1 - k, 2)), 1100 + k * 1500, .07); } pluck(N.Fs5, t + 1.12, .24); thump(t + 1.12, 60, .6); },
    pulse(t) { for (let i = 0; i < 3; i++) { thump(t + i * .9, 52, .7); thump(t + i * .9 + .16, 46, .4); } [N.Cs5, N.Fs5, N.Gs5, N.Cs6].forEach((f, i) => pluck(f, t + .15 + i * .32, .1)); },
    rows(t) { [N.Fs5, N.Ds4, N.Gs5, N.Cs6].forEach((f, i) => { const u = t + .25 + i * .13; tick(u, 1800, .07); pluck(f, u, i === 1 ? .12 : .17); }); thump(t + .06, 58, .5); },
    padOn(t) { if (pad) return; const g = ctx.createGain(), f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(300, t); f.frequency.linearRampToValueAtTime(1100, t + 12); g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(.07, t + 1.2);
      const os = [N.Fs2, N.Cs3, N.Gs3, N.Cs4].flatMap(n => [n * .997, n * 1.003]).map(fr => { const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = fr; o.connect(f); o.start(t); return o; });
      f.connect(g); out(g, .3); pad = {g, os}; },
    padOff(t) { if (!pad) return; const p = pad; pad = null; p.g.gain.cancelScheduledValues(t); p.g.gain.setValueAtTime(p.g.gain.value, t); p.g.gain.linearRampToValueAtTime(0.0001, t + .6); p.os.forEach(o => o.stop(t + .7)); }
  };
  window.Snd = {
    enabled: true,
    start() { init(); if (ctx && ctx.state === 'suspended') ctx.resume(); },
    cue(name, arg, delay) { if (!ctx || !this.enabled || !C[name]) return; try { C[name](now() + (delay || 0) + .01, arg); } catch (e) {} },
    stop() { if (ctx) C.padOff(now()); },
    notes: N
  };
})();
