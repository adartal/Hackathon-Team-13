import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Button, Chip, CircularProgress, Typography } from "@mui/material";
import PersonIcon from "@mui/icons-material/Person";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ForumIcon from "@mui/icons-material/Forum";
import SchoolIcon from "@mui/icons-material/School";
import AssignmentIcon from "@mui/icons-material/Assignment";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  getUser,
  getStudentOverview,
  getStudentAiSummary,
  type StudentOverview,
} from "@/lib/api";
import { AppHeader } from "@/components/AppHeader";
import {
  PageRoot,
  Main,
  HeaderCard,
  AvatarCircle,
  HeaderInfo,
  HeaderId,
  HeaderSub,
  SummaryCard,
  SummaryLabel,
  StatsRow,
  StatBox,
  StatValue,
  StatLabel,
  SectionTitle,
  ConvoList,
  ConvoItem,
  AiSummaryText,
  EmptyTldr,
} from "./students.$id.style";

export const Route = createFileRoute("/students/$id")({
  head: () => ({ meta: [{ title: "Student progress — Hintly" }] }),
  component: StudentStatusPage,
});

function buildStatsSummary(overview: StudentOverview): string {
  const { stats } = overview;
  if (stats.total_conversations === 0) return "No conversations yet.";
  const parts: string[] = [];
  if (stats.done_count > 0)
    parts.push(`${stats.done_count} submitted`);
  if (stats.assigned_count > 0)
    parts.push(`${stats.assigned_count} assigned`);
  if (stats.practice_count > 0)
    parts.push(`${stats.practice_count} practice`);
  return `${stats.total_conversations} session${stats.total_conversations !== 1 ? "s" : ""} (${parts.join(", ")}), ${stats.total_turns} turn${stats.total_turns !== 1 ? "s" : ""} total.`;
}

function StudentStatusPage() {
  const { id } = Route.useParams();
  const navigate = useNavigate();
  const [overview, setOverview] = useState<StudentOverview | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const user = getUser();
    if (!user) { navigate({ to: "/" }); return; }
    if (user.role !== "teacher") { navigate({ to: "/home" }); return; }
    getStudentOverview(user.id, id)
      .then(setOverview)
      .catch(() => setLoadError("Could not load student data."));
  }, [id, navigate]);

  function loadAiSummary() {
    const user = getUser();
    if (!user) return;
    setAiLoading(true);
    setAiError(false);
    getStudentAiSummary(user.id, id)
      .then((s) => { setAiSummary(s); setAiLoading(false); })
      .catch(() => { setAiError(true); setAiLoading(false); });
  }

  if (loadError) {
    return (
      <PageRoot>
        <AppHeader title="Student progress" back="/teacher" />
        <Main><Typography color="error">{loadError}</Typography></Main>
      </PageRoot>
    );
  }

  if (!overview) {
    return (
      <PageRoot>
        <AppHeader title="Student progress" back="/teacher" />
        <Main>
          <div style={{ display: "flex", justifyContent: "center", paddingTop: 40 }}>
            <CircularProgress />
          </div>
        </Main>
      </PageRoot>
    );
  }

  const { stats, conversations } = overview;
  const done = conversations.filter((c) => c.status === "completed");
  const activeAssigned = conversations.filter((c) => c.assigned_by && c.status !== "completed");
  const practice = conversations.filter((c) => !c.assigned_by && c.status !== "completed");

  function openConversation(conversationId: string) {
    navigate({
      to: "/review/$id",
      params: { id: conversationId },
      search: { studentId: overview!.student_id },
    });
  }

  return (
    <PageRoot>
      <AppHeader title="Student progress" back="/teacher" />
      <Main>
        {/* Header */}
        <HeaderCard>
          <AvatarCircle>
            <PersonIcon sx={{ fontSize: 28 }} />
          </AvatarCircle>
          <HeaderInfo>
            <HeaderId>{overview.username ?? id}</HeaderId>
            <HeaderSub>{buildStatsSummary(overview)}</HeaderSub>
          </HeaderInfo>
        </HeaderCard>

        {/* Stats */}
        <StatsRow>
          <StatBox>
            <StatValue>{stats.total_conversations}</StatValue>
            <StatLabel>Sessions</StatLabel>
          </StatBox>
          <StatBox done>
            <StatValue done>{stats.done_count}</StatValue>
            <StatLabel>Submitted</StatLabel>
          </StatBox>
          <StatBox>
            <StatValue>{stats.total_turns}</StatValue>
            <StatLabel>Turns</StatLabel>
          </StatBox>
        </StatsRow>

        {/* AI Summary */}
        <SummaryCard>
          <SummaryLabel>
            <AutoAwesomeIcon sx={{ fontSize: 16 }} />
            <span>AI progress summary</span>
            {!aiSummary && !aiLoading && (
              <Button
                size="small"
                variant="outlined"
                sx={{ ml: "auto", fontSize: "0.7rem", py: 0.25, px: 1, minWidth: 0 }}
                onClick={loadAiSummary}
              >
                Generate
              </Button>
            )}
            {aiSummary && !aiLoading && (
              <Button
                size="small"
                startIcon={<RefreshIcon sx={{ fontSize: 14 }} />}
                sx={{ ml: "auto", fontSize: "0.7rem", py: 0.25, px: 1, minWidth: 0 }}
                onClick={loadAiSummary}
              >
                Refresh
              </Button>
            )}
          </SummaryLabel>

          {aiLoading && (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Analysing conversations…
              </Typography>
            </div>
          )}
          {aiError && (
            <Typography variant="body2" color="error">
              Could not generate summary. Try again.
            </Typography>
          )}
          {aiSummary && !aiLoading && (
            <AiSummaryText>{aiSummary}</AiSummaryText>
          )}
          {!aiSummary && !aiLoading && !aiError && (
            <Typography variant="body2" color="text.secondary" sx={{ fontStyle: "italic" }}>
              Click "Generate" to get an AI analysis of this student's strengths and struggles.
            </Typography>
          )}
        </SummaryCard>

        {/* Done sessions */}
        {done.length > 0 && (
          <section>
            <SectionTitle>
              <CheckCircleIcon sx={{ fontSize: 20, color: "success.main" }} />
              <span>Submitted sessions</span>
            </SectionTitle>
            <ConvoList>
              {done.map((c) => (
                <ConvoItem key={c.id} onClick={() => openConversation(c.id)}>
                  {c.assigned_by
                    ? <SchoolIcon sx={{ fontSize: 20, color: "success.main", flexShrink: 0 }} />
                    : <ForumIcon sx={{ fontSize: 20, color: "success.main", flexShrink: 0 }} />}
                  <Typography variant="body2" fontWeight={500} sx={{ flex: 1 }}>
                    {c.name}
                  </Typography>
                  <Chip label="הושלם" size="small" color="success" variant="outlined" sx={{ fontSize: "0.65rem", height: 20 }} />
                  <ChevronRightIcon sx={{ fontSize: 18, color: "text.disabled" }} />
                </ConvoItem>
              ))}
            </ConvoList>
          </section>
        )}

        {/* Active assigned questions */}
        {activeAssigned.length > 0 && (
          <section>
            <SectionTitle>
              <AssignmentIcon sx={{ fontSize: 20, color: "primary.main" }} />
              <span>Assigned questions</span>
            </SectionTitle>
            <ConvoList>
              {activeAssigned.map((c) => (
                <ConvoItem key={c.id} onClick={() => openConversation(c.id)}>
                  <SchoolIcon sx={{ fontSize: 20, color: "primary.main", flexShrink: 0 }} />
                  <Typography variant="body2" fontWeight={500} sx={{ flex: 1 }}>
                    {c.name}
                  </Typography>
                  <Chip label="Assigned" size="small" color="primary" variant="outlined" sx={{ fontSize: "0.65rem", height: 20 }} />
                  <ChevronRightIcon sx={{ fontSize: 18, color: "text.disabled" }} />
                </ConvoItem>
              ))}
            </ConvoList>
          </section>
        )}

        {/* Practice sessions */}
        {practice.length > 0 && (
          <section>
            <SectionTitle>
              <ForumIcon sx={{ fontSize: 20, color: "text.secondary" }} />
              <span>Practice sessions</span>
            </SectionTitle>
            <ConvoList>
              {practice.map((c) => (
                <ConvoItem key={c.id} onClick={() => openConversation(c.id)}>
                  <ForumIcon sx={{ fontSize: 20, color: "text.secondary", flexShrink: 0 }} />
                  <Typography variant="body2" fontWeight={500} sx={{ flex: 1 }}>
                    {c.name}
                  </Typography>
                  <ChevronRightIcon sx={{ fontSize: 18, color: "text.disabled" }} />
                </ConvoItem>
              ))}
            </ConvoList>
          </section>
        )}

        {conversations.length === 0 && (
          <EmptyTldr>
            No conversations yet — assign a question or wait for the student to start practicing.
          </EmptyTldr>
        )}
      </Main>
    </PageRoot>
  );
}
