'use client';

import { useEffect, useState } from 'react';
import {
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { chatAPI, EvalRun } from '@/lib/api';

const METRIC_COLORS: Record<string, string> = {
  faithfulness: 'var(--chart-1)',
  answer_relevancy: 'var(--chart-2)',
  context_precision: 'var(--chart-3)',
  context_recall: 'var(--chart-4)',
};

const METRIC_LABELS: Record<string, string> = {
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Answer Relevancy',
  context_precision: 'Context Precision',
  context_recall: 'Context Recall',
};

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

function StatusPill({ passed }: { passed: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm font-medium ${
        passed
          ? 'border-[var(--chart-2)]/40 bg-[var(--chart-2)]/10 text-[var(--chart-2)]'
          : 'border-destructive/40 bg-destructive/10 text-destructive'
      }`}
    >
      <span
        className={`h-2 w-2 rounded-full ${passed ? 'bg-[var(--chart-2)]' : 'bg-destructive'}`}
      />
      {passed ? 'Passing' : 'Failing'}
    </span>
  );
}

function MetricCard({
  metricKey,
  score,
  threshold,
}: {
  metricKey: string;
  score: number;
  threshold: number | undefined;
}) {
  const passed = threshold === undefined || score >= threshold;
  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">
          {METRIC_LABELS[metricKey] ?? metricKey}
        </span>
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            passed ? 'bg-[var(--chart-2)]' : 'bg-destructive'
          }`}
        />
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-3xl font-semibold tabular-nums text-foreground">
          {(score * 100).toFixed(0)}
          <span className="text-lg text-muted-foreground">%</span>
        </span>
        {threshold !== undefined && (
          <span className="text-xs text-muted-foreground">
            threshold {(threshold * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

export default function EvalDashboardPage() {
  const [latest, setLatest] = useState<EvalRun | null | undefined>(undefined);
  const [history, setHistory] = useState<EvalRun[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [latestRun, historyResp] = await Promise.all([
          chatAPI.getLatestEvalRun(),
          chatAPI.getEvalRunHistory(50),
        ]);
        if (!cancelled) {
          setLatest(latestRun);
          // Oldest-first for the trend chart's left-to-right reading order
          setHistory([...historyResp.runs].reverse());
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load eval data');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loading = latest === undefined && !error;

  const chartData = history.map((run) => ({
    label: formatTimestamp(run.run_at),
    commit: run.commit_sha?.slice(0, 7) ?? 'unknown',
    ...Object.fromEntries(
      Object.entries(run.metric_averages).map(([k, v]) => [k, Number((v * 100).toFixed(1))])
    ),
  }));

  const metricKeys = latest?.metric_averages ? Object.keys(latest.metric_averages) : [];

  return (
    <div className="min-h-screen bg-background px-6 py-10 md:px-10">
      <div className="mx-auto max-w-5xl">
        <div className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-foreground">
              Evaluation Dashboard
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              RAG quality over time, scored by{' '}
              <a
                href="https://github.com/explodinggradients/ragas"
                target="_blank"
                rel="noreferrer"
                className="underline decoration-dotted underline-offset-2 hover:text-foreground"
              >
                Ragas
              </a>{' '}
              against a golden question set - see{' '}
              <code className="rounded bg-muted px-1 py-0.5 text-xs">eval/run_eval.py</code>.
            </p>
          </div>
          {latest && <StatusPill passed={latest.passed} />}
        </div>

        {loading && (
          <div className="rounded-xl border border-border bg-card p-10 text-center text-sm text-muted-foreground">
            Loading evaluation history…
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-6 text-sm text-destructive">
            Couldn&apos;t load evaluation data: {error}. Is the backend running?
          </div>
        )}

        {!loading && !error && latest === null && (
          <div className="rounded-xl border border-dashed border-border bg-card p-10 text-center">
            <p className="text-sm font-medium text-foreground">No evaluation runs recorded yet</p>
            <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
              Run the evaluation suite once to populate this dashboard:
            </p>
            <pre className="mx-auto mt-4 w-fit rounded-lg bg-muted px-4 py-2 text-left text-xs text-foreground">
              python -m eval.run_eval
            </pre>
          </div>
        )}

        {!loading && !error && latest && (
          <>
            <div className="mb-4 text-xs text-muted-foreground">
              Latest run: {formatTimestamp(latest.run_at)}
              {latest.commit_sha && (
                <>
                  {' '}
                  · commit <code className="rounded bg-muted px-1 py-0.5">{latest.commit_sha.slice(0, 7)}</code>
                </>
              )}
              {' '}· {latest.num_cases - latest.num_failed_cases}/{latest.num_cases} cases evaluated cleanly
            </div>

            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              {metricKeys.map((key) => (
                <MetricCard
                  key={key}
                  metricKey={key}
                  score={latest.metric_averages[key]}
                  threshold={latest.thresholds[key]}
                />
              ))}
            </div>

            {chartData.length > 1 && (
              <div className="mt-8 rounded-xl border border-border bg-card p-5">
                <h2 className="mb-4 text-sm font-medium text-foreground">Score trend across runs</h2>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="commit"
                      tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                      stroke="var(--border)"
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                      stroke="var(--border)"
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Legend
                      formatter={(value) => METRIC_LABELS[value] ?? value}
                      wrapperStyle={{ fontSize: 12 }}
                    />
                    {metricKeys.map((key) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={METRIC_COLORS[key] ?? 'var(--chart-5)'}
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="mt-8">
              <h2 className="mb-3 text-sm font-medium text-foreground">Recent runs</h2>
              <div className="overflow-hidden rounded-xl border border-border">
                <table className="w-full text-left text-sm">
                  <thead className="bg-muted/50 text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2 font-medium">Status</th>
                      <th className="px-4 py-2 font-medium">Run at</th>
                      <th className="px-4 py-2 font-medium">Commit</th>
                      <th className="px-4 py-2 font-medium">Cases</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...history].reverse().map((run) => (
                      <tr key={run.id} className="border-t border-border">
                        <td className="px-4 py-2">
                          <span
                            className={`inline-block h-2 w-2 rounded-full ${
                              run.passed ? 'bg-[var(--chart-2)]' : 'bg-destructive'
                            }`}
                          />
                        </td>
                        <td className="px-4 py-2 text-foreground">{formatTimestamp(run.run_at)}</td>
                        <td className="px-4 py-2 font-mono text-xs text-muted-foreground">
                          {run.commit_sha ? run.commit_sha.slice(0, 7) : '—'}
                        </td>
                        <td className="px-4 py-2 text-muted-foreground">
                          {run.num_cases - run.num_failed_cases}/{run.num_cases}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
