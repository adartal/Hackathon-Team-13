import { styled, alpha } from "@mui/material/styles";
import { Box, Paper } from "@mui/material";

export const PageRoot = styled(Box)({
  minHeight: "100vh",
});

export const Main = styled(Box)(({ theme }) => ({
  maxWidth: 672,
  margin: "0 auto",
  padding: theme.spacing(3, 2, 5),
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(3),
}));

export const HeaderCard = styled(Paper)(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  gap: theme.spacing(2),
  padding: theme.spacing(2.5),
  borderRadius: 20,
  border: `1px solid ${theme.palette.divider}`,
  boxShadow: "none",
}));

export const AvatarCircle = styled(Box)(({ theme }) => ({
  width: 56,
  height: 56,
  borderRadius: "50%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  flexShrink: 0,
  backgroundColor: alpha(theme.palette.primary.main, 0.12),
  color: theme.palette.primary.main,
}));

export const HeaderInfo = styled(Box)({
  minWidth: 0,
});

export const HeaderId = styled("div")({
  fontWeight: 700,
  fontSize: "1.4rem",
  lineHeight: 1.2,
});

export const HeaderSub = styled("div")(({ theme }) => ({
  fontSize: "0.8rem",
  color: theme.palette.text.secondary,
  marginTop: 2,
}));

export const SummaryCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2.5),
  borderRadius: 20,
  border: `1px solid ${theme.palette.divider}`,
  boxShadow: "none",
}));

export const SummaryLabel = styled(Box)(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  gap: theme.spacing(0.75),
  color: theme.palette.primary.main,
  fontWeight: 600,
  fontSize: "0.75rem",
  textTransform: "uppercase",
  letterSpacing: 0.4,
  marginBottom: theme.spacing(1.5),
}));

export const StatsRow = styled(Box)(({ theme }) => ({
  display: "grid",
  gridTemplateColumns: "repeat(3, 1fr)",
  gap: theme.spacing(1.5),
}));

export const StatBox = styled(Box, {
  shouldForwardProp: (p) => p !== "done",
})<{ done?: boolean }>(({ theme, done }) => ({
  textAlign: "center",
  padding: theme.spacing(1.5),
  borderRadius: 14,
  backgroundColor: done
    ? alpha(theme.palette.success.main, 0.08)
    : alpha(theme.palette.primary.main, 0.06),
  border: `1px solid ${done
    ? alpha(theme.palette.success.main, 0.25)
    : alpha(theme.palette.primary.main, 0.15)}`,
}));

export const StatValue = styled("div", {
  shouldForwardProp: (p) => p !== "done",
})<{ done?: boolean }>(({ theme, done }) => ({
  fontWeight: 700,
  fontSize: "1.6rem",
  color: done ? theme.palette.success.main : theme.palette.primary.main,
  lineHeight: 1,
}));

export const StatLabel = styled("div")(({ theme }) => ({
  fontSize: "0.72rem",
  color: theme.palette.text.secondary,
  marginTop: 4,
  fontWeight: 500,
}));

export const SectionTitle = styled(Box)(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  gap: theme.spacing(1),
  fontWeight: 600,
  fontSize: "0.95rem",
  marginBottom: theme.spacing(1.5),
}));

export const ConvoList = styled(Box)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(1),
}));

export const ConvoItem = styled(Paper)(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  gap: theme.spacing(1.5),
  padding: theme.spacing(1.5, 2),
  borderRadius: 14,
  border: `1px solid ${theme.palette.divider}`,
  boxShadow: "none",
  cursor: "pointer",
  transition: "box-shadow 150ms ease, border-color 150ms ease",
  "&:hover": {
    boxShadow: theme.shadows[2],
    borderColor: theme.palette.primary.light,
  },
}));

export const AiSummaryText = styled("p")(({ theme }) => ({
  margin: 0,
  fontSize: "0.9rem",
  lineHeight: 1.6,
  color: theme.palette.text.primary,
}));

export const EmptyTldr = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: 14,
  border: `1px dashed ${theme.palette.divider}`,
  textAlign: "center",
  fontSize: "0.85rem",
  color: theme.palette.text.secondary,
}));
