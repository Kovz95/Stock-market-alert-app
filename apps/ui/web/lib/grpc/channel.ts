import { createChannel, createClientFactory } from "nice-grpc";
import {
  AlertServiceDefinition,
  type AlertServiceClient,
} from "@gen/alert/v1/alert";

const GRPC_ENDPOINT = process.env.GRPC_ENDPOINT || "localhost:8080";

const channel = createChannel(GRPC_ENDPOINT);

const clientFactory = createClientFactory();

export const alertClient: AlertServiceClient = clientFactory.create(
  AlertServiceDefinition,
  channel
);
