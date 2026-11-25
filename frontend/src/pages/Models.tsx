import { Award, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Badge, Card, ErrorState, Loading, PageHeader, SourceTag, Tabs } from '../components/ui'
import { api } from '../lib/api'
import { bytes, COLORS, MODEL_COLORS, num, pct, shortParam } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { Confusion, MetricsResponse, ModelsResponse, RocResponse, SweepPoint } from '../lib/types'

export function ConfusionMatrix({ cm, title }: { cm: Confusion; title?: string }) {
  const cells = [
    { k: 'TN', v: cm.tn, good: true, label: 'true benign' },
    { k: 'FP', v: cm.fp, good: false, label: 'false malignant' },
    { k: 'FN', v: cm.fn, good: false, label: 'missed malignant' },
    { k: 'TP', v: cm.tp, good: true, label: 'true malignant' },
  ]
  return (
    <figure>
      {title && <figcaption className="mb-1 truncate text-xs font-medium text-slate-700">{title}</figcaption>}
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center text-xs">
        <div />
        <div className="text-[10px] text-slate-500">pred. benign</div>
        <div className="text-[10px] text-slate-500">pred. malignant</div>
        {[0, 1].map((r) => (
          <div key={r} className="contents">
            <div className="flex items-center pr-1 text-[10px] text-slate-500">{r === 0 ? 'benign' : 'malignant'}</div>
            {cells.slice(r * 2, r * 2 + 2).map((c) => (
              <div key={c.k} title={`${c.k}: ${c.label}`} className={`rounded-md py-2 ${c.good ? 'bg-teal-50 text-teal-900' : c.v > 0 ? 'bg-amber-50 text-amber-900' : 'bg-slate-50 text-slate-500'}`}>
                <div className="text-lg font-semibold tabular-nums">{c.v}</div>
                <div className="text-[10px]">{c.k}</div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </figure>
  )
}

function ComparisonTable({ models, metrics }: { models: ModelsResponse; metrics: MetricsResponse }) {
  return (
    <Card title="Algorithm comparison" subtitle="CV = mean ± SD over outer folds of nested CV (training split, threshold 0.5). Test = single evaluation on the held-out split (threshold 0.5)." action={<div className="flex gap-1"><SourceTag kind="cv" /><SourceTag kind="test" /></div>}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-slate-500">
            <tr className="border-b">
              <th className="py-2 pr-3 font-medium">Model</th>
              <th className="bg-teal-50/60 px-2 font-medium">CV AUC</th>
              <th className="bg-teal-50/60 px-2 font-medium">CV recall</th>
              <th className="bg-teal-50/60 px-2 font-medium">CV specificity</th>
              <th className="bg-teal-50/60 px-2 font-medium">CV F1</th>
              <th className="bg-teal-50/60 px-2 font-medium">CV Brier</th>
              <th className="bg-violet-50/60 px-2 font-medium">Test acc.</th>
              <th className="bg-violet-50/60 px-2 font-medium">Test recall</th>
              <th className="bg-violet-50/60 px-2 font-medium">Test spec.</th>
              <th className="bg-violet-50/60 px-2 font-medium">Test prec.</th>
              <th className="bg-violet-50/60 px-2 font-medium">Test F1</th>
              <th className="bg-violet-50/60 px-2 font-medium">Test AUC</th>
              <th className="px-2 font-medium">Fit (s)</th>
              <th className="px-2 font-medium">Infer (ms/row)</th>
              <th className="px-2 font-medium">Size</th>
              <th className="px-2 font-medium">Selected hyperparameters</th>
            </tr>
          </thead>
          <tbody>
            {models.models.map((m) => {
              const mm = metrics.models.find((x) => x.key === m.key)
              if (m.status === 'failed' || !mm?.cv || !mm.test) {
                return (
                  <tr key={m.key} className="border-b border-slate-100">
                    <td className="py-2 pr-3 font-medium">{m.name}</td>
                    <td colSpan={15} className="px-2 text-amber-800"><XCircle className="mr-1 inline h-3.5 w-3.5" aria-hidden />Experiment failed: {m.error ?? mm?.error ?? 'unknown error'}</td>
                  </tr>
                )
              }
              const cv = mm.cv
              const t = mm.test
              const pm = (x: { mean: number; std: number }) => `${num(x.mean, 3)} ± ${num(x.std, 3)}`
              return (
                <tr key={m.key} className={`border-b border-slate-100 ${m.selected ? 'bg-teal-50/40' : ''}`}>
                  <td className="whitespace-nowrap py-2 pr-3 font-medium text-navy-900">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: MODEL_COLORS[m.key] }} aria-hidden />
                    {m.name} {m.selected && <Badge tone="teal"><Award className="h-3 w-3" aria-hidden />selected</Badge>}
                  </td>
                  <td className="whitespace-nowrap px-2 tabular-nums">{pm(cv.roc_auc)}</td>
                  <td className="whitespace-nowrap px-2 tabular-nums">{pm(cv.recall)}</td>
                  <td className="whitespace-nowrap px-2 tabular-nums">{pm(cv.specificity)}</td>
                  <td className="whitespace-nowrap px-2 tabular-nums">{pm(cv.f1)}</td>
                  <td className="whitespace-nowrap px-2 tabular-nums">{num(cv.brier.mean, 4)}</td>
                  <td className="px-2 tabular-nums">{num(t.accuracy, 3)}</td>
                  <td className="px-2 tabular-nums">{num(t.recall, 3)}</td>
                  <td className="px-2 tabular-nums">{num(t.specificity, 3)}</td>
                  <td className="px-2 tabular-nums">{num(t.precision, 3)}</td>
                  <td className="px-2 tabular-nums">{num(t.f1, 3)}</td>
                  <td className="px-2 tabular-nums">{num(t.roc_auc, 3)}</td>
                  <td className="px-2 tabular-nums">{num(m.training_seconds, 3)}</td>
                  <td className="px-2 tabular-nums">{num(m.inference?.batch_per_row_ms, 4)}</td>
                  <td className="whitespace-nowrap px-2 tabular-nums">{bytes(m.model_size_bytes)}</td>
                  <td className="px-2 font-mono text-[10px] text-slate-600">
                    {Object.entries(m.best_params ?? {}).map(([k, v]) => `${shortParam(k)}=${v}`).join(', ')}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[11px] text-slate-500">
        Fit = refit time of the tuned pipeline on the full training split. Inference = predict_proba time per row on a 114-row batch. Test results for non-selected models are shown for transparency and were not used to choose the model.
      </p>
    </Card>
  )
}

function CurveChart({ roc, visible, kind }: { roc: RocResponse; visible: Set<string>; kind: 'roc' | 'pr' | 'calibration' }) {
  const cfg = {
    roc: { x: 'fpr', y: 'tpr', xl: 'False positive rate', yl: 'Recall (TPR)' },
    pr: { x: 'recall', y: 'precision', xl: 'Recall', yl: 'Precision' },
    calibration: { x: 'mean_predicted', y: 'fraction_positive', xl: 'Mean predicted P(malignant)', yl: 'Observed fraction malignant' },
  }[kind]
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 8, right: 16, bottom: 20, left: 0 }}>
          <CartesianGrid stroke="#e2e8f0" />
          <XAxis type="number" dataKey={cfg.x} domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: cfg.xl, position: 'insideBottom', offset: -10, fontSize: 11 }} />
          <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: cfg.yl, angle: -90, position: 'insideLeft', fontSize: 11, dy: 50 }} />
          <Tooltip formatter={(v) => Number(v).toFixed(3)} labelFormatter={(l) => `${cfg.xl}: ${Number(l).toFixed(3)}`} />
          {kind !== 'pr' && (
            <Line data={[{ [cfg.x]: 0, [cfg.y]: 0 }, { [cfg.x]: 1, [cfg.y]: 1 }]} dataKey={cfg.y} stroke={COLORS.slate} strokeDasharray="4 4" dot={false} name={kind === 'roc' ? 'chance' : 'perfect calibration'} isAnimationActive={false} />
          )}
          {roc.models.filter((m) => m.curves && visible.has(m.key)).map((m) => (
            <Line
              key={m.key}
              data={m.curves![kind]}
              dataKey={cfg.y}
              name={m.name}
              stroke={MODEL_COLORS[m.key]}
              strokeWidth={m.key === roc.selected_key ? 2.5 : 1.25}
              dot={kind === 'calibration'}
              type={kind === 'pr' ? 'stepAfter' : 'linear'}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function ThresholdExplorer({ roc, operating }: { roc: RocResponse; operating: number }) {
  const ok = roc.models.filter((m) => m.threshold_sweep)
  const [key, setKey] = useState(roc.selected_key)
  const [source, setSource] = useState<'test' | 'oof'>('oof')
  const [t, setT] = useState(Math.round(operating * 100) / 100)
  const sweep: SweepPoint[] =
    source === 'oof' ? roc.oof_threshold_sweep : (ok.find((m) => m.key === key)?.threshold_sweep ?? [])
  const point = sweep.reduce((best, p) => (Math.abs(p.threshold - t) < Math.abs(best.threshold - t) ? p : best), sweep[0])
  return (
    <Card
      title="Threshold explorer"
      subtitle="Lowering the threshold catches more malignant samples (higher recall) at the cost of more false positives."
      action={source === 'oof' ? <SourceTag kind="oof" /> : <SourceTag kind="test" />}
    >
      <div className="mb-4 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex min-w-0 max-w-full items-center gap-2">Data
          <select value={source} onChange={(e) => setSource(e.target.value as 'test' | 'oof')} className="min-w-0 max-w-full rounded-md border border-slate-300 px-2 py-1 text-sm">
            <option value="oof">Selected model · out-of-fold training predictions</option>
            <option value="test">Held-out test predictions</option>
          </select>
        </label>
        {source === 'test' && (
          <label className="flex min-w-0 max-w-full items-center gap-2">Model
            <select value={key} onChange={(e) => setKey(e.target.value)} className="rounded-md border border-slate-300 px-2 py-1 text-sm">
              {ok.map((m) => <option key={m.key} value={m.key}>{m.name}</option>)}
            </select>
          </label>
        )}
        <label className="flex flex-1 items-center gap-2">
          <span className="whitespace-nowrap">Threshold <span className="font-mono tabular-nums">{t.toFixed(2)}</span></span>
          <input aria-label="Classification threshold" type="range" min={0.01} max={0.99} step={0.01} value={t} onChange={(e) => setT(Number(e.target.value))} className="w-full accent-teal-600" />
        </label>
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="h-64 lg:col-span-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sweep} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="#e2e8f0" />
              <XAxis dataKey="threshold" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => Number(v).toFixed(3)} />
              <Legend />
              <ReferenceLine x={t} stroke={COLORS.navy} />
              <ReferenceLine x={operating} stroke={COLORS.teal} strokeDasharray="4 4" label={{ value: 'operating', fontSize: 10, fill: COLORS.teal, position: 'insideTopRight' }} />
              <Line dataKey="recall" name="recall (sensitivity)" stroke={COLORS.purple} dot={false} isAnimationActive={false} />
              <Line dataKey="specificity" stroke={COLORS.blue} dot={false} isAnimationActive={false} />
              <Line dataKey="precision" stroke={COLORS.slate} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        {point && (
          <div className="space-y-3">
            <ConfusionMatrix cm={point} title={`At threshold ${point.threshold.toFixed(2)}`} />
            <dl className="grid grid-cols-2 gap-1 text-xs">
              <dt className="text-slate-500">Recall</dt><dd className="tabular-nums">{pct(point.recall)}</dd>
              <dt className="text-slate-500">Specificity</dt><dd className="tabular-nums">{pct(point.specificity)}</dd>
              <dt className="text-slate-500">Precision</dt><dd className="tabular-nums">{pct(point.precision)}</dd>
              <dt className="text-slate-500">F1</dt><dd className="tabular-nums">{num(point.f1)}</dd>
            </dl>
            {source === 'test' && <p className="text-[11px] text-slate-500">Exploration only. The operating threshold was fixed from training out-of-fold predictions, never from this view.</p>}
          </div>
        )}
      </div>
    </Card>
  )
}

export default function Models() {
  const models = useApi(api.models)
  const metrics = useApi(api.metrics)
  const roc = useApi(api.roc)
  const [curve, setCurve] = useState<'roc' | 'pr' | 'calibration'>('roc')
  const [visible, setVisible] = useState<Set<string> | null>(null)

  const visibleSet = useMemo(
    () => visible ?? new Set(models.data?.models.filter((m) => m.status === 'completed').map((m) => m.key)),
    [visible, models.data],
  )

  if (models.loading || metrics.loading || roc.loading) return <><PageHeader title="Model Performance" /><Loading /></>
  const err = models.error ?? metrics.error ?? roc.error
  if (err) return <><PageHeader title="Model Performance" /><ErrorState error={err} onRetry={() => { models.reload(); metrics.reload(); roc.reload() }} /></>
  const M = models.data!
  const MT = metrics.data!
  const R = roc.data!
  const sel = MT.models.find((m) => m.key === MT.selected_key)!
  const selInfo = M.models.find((m) => m.selected)!
  const op = sel.test_at_operating_threshold!
  const opCi = sel.test_at_operating_threshold_ci95!
  const ok = MT.models.filter((m) => m.cv && m.test)

  const cvBars = ok.map((m) => ({
    name: m.name.split(' (')[0],
    recall: m.cv!.recall.mean, recallErr: m.cv!.recall.std,
    specificity: m.cv!.specificity.mean, specErr: m.cv!.specificity.std,
    auc: m.cv!.roc_auc.mean, aucErr: m.cv!.roc_auc.std,
  }))
  const testBars = ok.map((m) => ({ name: m.name.split(' (')[0], recall: m.test!.recall, specificity: m.test!.specificity }))
  const infer = M.models.filter((m) => m.inference).map((m) => ({ name: m.name.split(' (')[0], ms: m.inference!.single_row_ms_median, key: m.key }))

  const toggle = (k: string) => {
    const s = new Set(visibleSet)
    if (s.has(k)) s.delete(k)
    else s.add(k)
    setVisible(s)
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Model Performance & Comparison Lab" subtitle={<>All values below were produced by the training run <span className="font-mono">{M.model_version}</span>. Positive class = malignant.</>} />

      <Card title={<span className="flex items-center gap-2"><Award className="h-4 w-4 text-teal-600" aria-hidden /> Selected model: {selInfo.name}</span>}>
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-2 text-sm text-slate-700 lg:col-span-2">
            <p><span className="font-medium">Why this model?</span> {M.selection_rationale}</p>
            <p><span className="font-medium">Calibration:</span> {M.calibration.rule} Out-of-fold Brier uncalibrated {num(M.calibration.oof_brier_uncalibrated, 4)} vs sigmoid {num(M.calibration.oof_brier_sigmoid_calibrated, 4)} → applied: {M.calibration.applied}.</p>
            <p><span className="font-medium">Operating threshold {M.threshold.operating_threshold}:</span> {M.threshold.rule} Achieved OOF recall {pct(M.threshold.achieved_oof_recall)}, specificity {pct(M.threshold.achieved_oof_specificity)}.</p>
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between"><span className="text-xs font-medium text-slate-700">Test set @ threshold {op.threshold}</span><SourceTag kind="test" /></div>
            <ConfusionMatrix cm={op.confusion_matrix} />
            <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
              {(['recall', 'specificity', 'precision', 'accuracy', 'f1', 'roc_auc', 'brier'] as const).map((k) => (
                <div key={k} className="contents">
                  <dt className="text-slate-500">{k === 'roc_auc' ? 'ROC-AUC' : k}</dt>
                  <dd className="tabular-nums">{num(op[k], 3)} <span className="text-slate-400">[{num(opCi[k][0], 3)}, {num(opCi[k][1], 3)}]</span></dd>
                </div>
              ))}
            </dl>
            <p className="mt-2 text-[11px] text-slate-500">95% percentile bootstrap CIs (1000 resamples of {op.confusion_matrix.tn + op.confusion_matrix.fp + op.confusion_matrix.fn + op.confusion_matrix.tp} test samples). Small test set → wide intervals.</p>
          </div>
        </div>
      </Card>

      <ComparisonTable models={M} metrics={MT} />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Sensitivity vs specificity (CV)" subtitle="Mean ± SD across outer folds, threshold 0.5" action={<SourceTag kind="cv" />}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cvBars} margin={{ bottom: 30 }}>
                <CartesianGrid vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" interval={0} />
                <YAxis domain={[0.8, 1]} allowDataOverflow ticks={[0.8, 0.85, 0.9, 0.95, 1]} tickFormatter={(v: number) => v.toFixed(2)} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Legend verticalAlign="top" />
                <Bar dataKey="recall" name="recall" fill={COLORS.purple} isAnimationActive={false}><ErrorBar dataKey="recallErr" stroke="#334155" width={3} /></Bar>
                <Bar dataKey="specificity" fill={COLORS.blue} isAnimationActive={false}><ErrorBar dataKey="specErr" stroke="#334155" width={3} /></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Sensitivity vs specificity (test)" subtitle="Held-out split, threshold 0.5" action={<SourceTag kind="test" />}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={testBars} margin={{ bottom: 30 }}>
                <CartesianGrid vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" interval={0} />
                <YAxis domain={[0.8, 1]} allowDataOverflow ticks={[0.8, 0.85, 0.9, 0.95, 1]} tickFormatter={(v: number) => v.toFixed(2)} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Legend verticalAlign="top" />
                <Bar dataKey="recall" fill={COLORS.purple} isAnimationActive={false} />
                <Bar dataKey="specificity" fill={COLORS.blue} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Curves" subtitle="Held-out test predictions. Toggle models below." action={<SourceTag kind="test" />}>
        <Tabs label="Curve type" value={curve} onChange={setCurve} tabs={[{ id: 'roc', label: 'ROC' }, { id: 'pr', label: 'Precision–recall' }, { id: 'calibration', label: 'Calibration' }]} />
        <div className="mb-3 flex flex-wrap gap-2">
          {R.models.filter((m) => m.curves).map((m) => (
            <label key={m.key} className="flex cursor-pointer items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-xs">
              <input type="checkbox" checked={visibleSet.has(m.key)} onChange={() => toggle(m.key)} className="accent-teal-600" />
              <span className="h-2 w-2 rounded-full" style={{ background: MODEL_COLORS[m.key] }} aria-hidden />
              {m.name}
            </label>
          ))}
        </div>
        <CurveChart roc={R} visible={visibleSet} kind={curve} />
        {curve === 'calibration' && <p className="mt-2 text-xs text-slate-500">8 quantile bins over only 114 test samples — read as indicative. Brier scores are in the table above.</p>}
      </Card>

      <ThresholdExplorer roc={R} operating={MT.operating_threshold} />

      <Card title="Confusion matrices" subtitle="Held-out test set, threshold 0.5 for every model" action={<SourceTag kind="test" />}>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {ok.map((m) => <ConfusionMatrix key={m.key} cm={m.test!.confusion_matrix} title={m.name} />)}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="CV ROC-AUC per model" subtitle="Mean ± SD across outer folds" action={<SourceTag kind="cv" />}>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cvBars} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" domain={[0.9, 1]} allowDataOverflow ticks={[0.9, 0.925, 0.95, 0.975, 1]} tickFormatter={(v: number) => v.toFixed(3)} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                <Bar dataKey="auc" fill={COLORS.teal} isAnimationActive={false}><ErrorBar dataKey="aucErr" direction="x" stroke="#334155" width={3} /></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Inference latency" subtitle="Median single-row predict_proba time (ms), measured during training">
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={infer} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${Number(v).toFixed(3)} ms`} />
                <Bar dataKey="ms" fill={COLORS.navy} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Learning curve (selected model)" subtitle="ROC-AUC vs training size, 5-fold CV on the training split" action={<SourceTag kind="train" />}>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={MT.learning_curve}>
                <CartesianGrid stroke="#e2e8f0" />
                <XAxis dataKey="train_size" tick={{ fontSize: 11 }} />
                <YAxis domain={['auto', 1]} tick={{ fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(3)} />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                <Legend />
                <Line dataKey="train_auc_mean" name="training" stroke={COLORS.navy} isAnimationActive={false} />
                <Line dataKey="val_auc_mean" name="validation" stroke={COLORS.teal} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Error analysis (selected model)" subtitle={`Misclassified held-out samples at threshold ${MT.operating_threshold}`} action={<SourceTag kind="test" />}>
          {MT.selected_test_errors.length === 0 ? (
            <p className="text-sm text-slate-600">No misclassifications on this split — with 114 samples this does not imply error-free behaviour.</p>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="border-b text-slate-500"><tr><th className="py-1.5">Sample</th><th>Type</th><th>True</th><th>Predicted</th><th>P(malignant)</th></tr></thead>
              <tbody>
                {MT.selected_test_errors.map((e) => (
                  <tr key={e.sample_id} className="border-b border-slate-100">
                    <td className="py-1.5 font-mono">{e.sample_id}</td>
                    <td><Badge tone="amber">{e.error_type.replace('_', ' ')}</Badge></td>
                    <td>{e.true_label}</td><td>{e.predicted_label}</td>
                    <td className="tabular-nums">{num(e.probability_malignant, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-3 text-xs text-slate-500">False negatives (missed malignant samples) are the costliest error type in a screening context; they are reported explicitly rather than hidden inside accuracy.</p>
        </Card>
      </div>

      <Card title="Static evaluation figures" subtitle="Generated by scripts/train_models.py">
        <div className="grid gap-4 md:grid-cols-3">
          {['roc_test.png', 'calibration_test.png', 'cv_auc_boxplot.png'].map((f) => (
            <img key={f} src={api.figureUrl(f)} alt={f.replace(/_/g, ' ').replace('.png', '')} loading="lazy" className="w-full rounded border border-slate-200" />
          ))}
        </div>
      </Card>
    </div>
  )
}
