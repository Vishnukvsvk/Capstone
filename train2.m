dataDirectory = "mycnn";
percentTrainingSamples = 80;
percentValidationSamples = 10;
percentTestSamples = 10;
modulationTypes = categorical(["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM","1024QAM","PAM4", "GFSK", "CPFSK","DSB-AM","SSB-AM","B-FM"]);
sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
symbolsPerFrame = spf / sps;

%%%%%%%%%%%

frameDS = signalDatastore(dataDirectory,'SignalVariableNames',["frame","label"]);
frameDSTrans = transform(frameDS,@helperModClassIQAsPages);

splitPercentages = [percentTrainingSamples,percentValidationSamples,percentTestSamples];
[trainDSTrans,validDSTrans,testDSTrans] = helperModClassSplitData(frameDSTrans,splitPercentages);

trainFramesTall = tall(transform(trainDSTrans, @helperModClassReadFrame));
rxTrainFrames = gather(trainFramesTall);

rxTrainFrames = cat(4, rxTrainFrames{:});
validFramesTall = tall(transform(validDSTrans, @helperModClassReadFrame));
rxValidFrames = gather(validFramesTall);

rxValidFrames = cat(4, rxValidFrames{:});

% Gather the training and validation labels into the memory
trainLabelsTall = tall(transform(trainDSTrans, @helperModClassReadLabel));
rxTrainLabels = gather(trainLabelsTall);

validLabelsTall = tall(transform(validDSTrans, @helperModClassReadLabel));
rxValidLabels = gather(validLabelsTall);

modClassNet = helperModClassCNN(modulationTypes,sps,spf);

maxEpochs = 12;
miniBatchSize = 256;
options = helperModClassTrainingOptions(maxEpochs,miniBatchSize,...
  numel(rxTrainLabels),rxValidFrames,rxValidLabels);


fprintf('%s - Training the network\n', datestr(toc/86400,'HH:MM:SS'))
[trainedNet,info] = trainNetwork(rxTrainFrames,rxTrainLabels,modClassNet,options);
